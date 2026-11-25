"""Public surface for the anomaly baseline package."""

from .detect import (
    anomaly_batch,
    baseline,
    is_anomaly,
    median,
    quantile,
    update_baseline,
    validate_config,
    window_of,
)

__all__ = [
    "window_of",
    "median",
    "quantile",
    "baseline",
    "is_anomaly",
    "update_baseline",
    "validate_config",
    "anomaly_batch",
]