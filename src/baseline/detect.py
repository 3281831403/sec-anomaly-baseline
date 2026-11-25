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
    if float(size) <= 0.0:
        return []

    now = float(now)
    lower = now - float(size)
    points = [
        (float(timestamp), float(value))
        for timestamp, value in history
        if lower <= float(timestamp) < now
    ]
    points.sort(key=lambda point: point[0])
    return points


def median(values: Iterable[float]) -> float:
    """Return the median of a non-empty sequence of numbers."""
    data = sorted(float(item) for item in values)
    if not data:
        raise ValueError("median requires at least one value")
    middle = len(data) // 2
    if len(data) % 2:
        return data[middle]
    return (data[middle - 1] + data[middle]) / 2.0


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
    position = q * (len(data) - 1)
    lower = int(floor(position))
    upper = min(lower + 1, len(data) - 1)
    fraction = position - lower
    return data[lower] + (data[upper] - data[lower]) * fraction


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
    A value equal to the threshold is a deviation.  A baseline that carries a
    zero sample count has no reference and cannot flag a deviation.
    """
    if "count" in base and int(base["count"]) <= 0:
        return False
    factor = float(policy.get("factor", 1.5))
    upper = float(base.get("upper", 0.0))
    threshold = upper * factor
    return bool(value >= threshold)


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
    value = float(value)
    refresh = max(1, int(policy.get("refresh", 1)))
    freeze = int(policy.get("freeze", 0) or 0)

    freeze_left = int(result.get("freeze_left", 0))
    if freeze_left > 0:
        result["freeze_left"] = freeze_left - 1
        return result

    has_reference = "count" not in result or int(result["count"]) > 0
    if has_reference and is_anomaly(value, result, policy):
        result["freeze_left"] = freeze
        result["pending"] = 0
        return result

    pending = int(result.get("pending", 0)) + 1
    if pending < refresh:
        result["pending"] = pending
        return result

    result["pending"] = 0
    result["freeze_left"] = 0

    old_median = float(result.get("median", 0.0))
    old_upper = float(result.get("upper", 0.0))

    max_step = policy.get("max_step")
    step = None if max_step is None else max(0.0, float(max_step))

    median_value = value if step is None else min(value, old_median + step)
    upper_value = value if step is None else min(value, old_upper + step)

    result["median"] = median_value
    result["upper"] = max(old_upper, upper_value, median_value)
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
        if factor_value <= 1.0:
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

    for name, history in entities.items():
        records = list(history)
        current = _value_at(records, now)
        if current is None:
            continue
        reference = baseline(records, now, window)
        result[name] = is_anomaly(current, reference, policy)

    return result
