"""Robust, per-entity anomaly baselines over time windows.

A history is a sequence of ``(timestamp, value)`` pairs.  Timestamps and
window sizes are expressed in seconds.  The baseline for an observation at
``now`` is derived only from strictly earlier samples inside the configured
window, so the observation being scored never contributes to its own
reference.
"""

from __future__ import annotations

from math import floor
from typing import Any, Iterable, Mapping, Sequence


History = Iterable[Sequence[float]]
Point = tuple[float, float]


def window_of(history: History, now: float, size: float) -> list[Point]:
    """Return the samples in the historical window ending at ``now``.

    The window spans ``size`` seconds and is closed on the left and open on
    the right.  Samples are returned in ascending timestamp order and the
    sample at ``now`` is not part of the window.
    """
    count = int(size)
    if count <= 0:
        return []
    return list(history)[-count:]


def median(values: Iterable[float]) -> float:
    """Return the median of a non-empty sequence of numbers."""
    data = [float(item) for item in values]
    if not data:
        raise ValueError("median requires at least one value")
    return sum(data) / len(data)


def quantile(values: Iterable[float], q: float) -> float:
    """Return the ``q`` quantile using linear interpolation.

    ``q`` must be between zero and one inclusive.  The endpoints map to the
    smallest and largest observations respectively.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile requires 0 <= q <= 1")
    data = sorted(float(item) for item in values)
    if not data:
        raise ValueError("quantile requires at least one value")
    index = min(int(q * (len(data) - 1)), len(data) - 1)
    return data[index]


def baseline(history: History, now: float, size: float) -> dict[str, float | int]:
    """Compute the robust baseline for the observation at ``now``.

    The returned mapping contains the historical median, the ninety-fifth
    percentile of the historical values, and the number of samples used.
    """
    window = window_of(history, now, size)
    values = [float(value) for _, value in window]
    if not values:
        return {"median": 0.0, "upper": 0.0, "count": 0}
    return {
        "median": median(values),
        "upper": quantile(values, 0.95),
        "count": len(values),
    }


def is_anomaly(
    value: float,
    base: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> bool:
    """Return whether ``value`` reaches the policy threshold for ``base``.

    The threshold is the configured multiple of the baseline upper quantile.
    A value equal to the threshold is a deviation.
    """
    factor = float(policy.get("factor", 1.5))
    upper = float(base.get("upper", 0.0))
    threshold = upper * factor
    return bool(value > threshold)


def update_baseline(
    base: Mapping[str, Any],
    value: float,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Advance a baseline with one observation.

    ``refresh`` observations are accumulated before the statistics move.
    ``freeze`` is the number of observations held after an anomalous sample.
    ``max_step`` bounds an upward move so gradual drift cannot lift the
    baseline without limit.  An anomalous observation never becomes part of
    the reference statistics.
    """
    result = dict(base)
    refresh = max(1, int(policy.get("refresh", 1)))
    pending = int(result.get("pending", 0)) + 1
    if pending < refresh:
        result["pending"] = pending
        return result

    result["pending"] = 0
    old_upper = float(result.get("upper", 0.0))
    result["median"] = float(value)
    result["upper"] = max(old_upper, float(value))
    result["freeze_left"] = int(result.get("freeze_left", 0))
    return result


def validate_config(policy: Mapping[str, Any]) -> list[str]:
    """Return a list of human-readable policy problems."""
    issues: list[str] = []

    factor = policy.get("factor", 1.5)
    try:
        factor_value = float(factor)
    except (TypeError, ValueError):
        issues.append("factor must be a number greater than one")
    else:
        if factor_value == 1.0:
            issues.append("factor must be greater than one")

    freeze = policy.get("freeze")
    try:
        freeze_value = int(freeze)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        issues.append("freeze period must be configured")
    else:
        if freeze_value <= 0:
            issues.append("freeze period must be configured")

    return issues


def _value_at(history: History, now: float) -> float | None:
    """Return the observation recorded exactly at ``now``, if present."""
    for timestamp, value in reversed(list(history)):
        if float(timestamp) == float(now):
            return float(value)
    return None


def anomaly_batch(
    entities: Mapping[str, History],
    policy: Mapping[str, Any],
    now: float,
) -> dict[str, bool]:
    """Score the current observation of every entity independently.

    ``entities`` maps an entity name to its history.  The current observation
    is the record whose timestamp equals ``now``.  Each entity uses its own
    baseline; nothing is shared across the batch.
    """
    result: dict[str, bool] = {}
    window = float(policy.get("window", 0.0))
    reference: dict[str, Any] | None = None

    for name, history in entities.items():
        if reference is None:
            reference = baseline(history, now, window)
        current = _value_at(history, now)
        if current is None:
            continue
        result[name] = is_anomaly(current, reference, policy)

    return result