from baseline.detect import (
    anomaly_batch,
    is_anomaly,
    median,
    update_baseline,
    window_of,
)


def test_window_excludes_current_point():
    history = [(0.0, 1.0), (10.0, 2.0), (20.0, 3.0)]

    assert window_of(history, now=20.0, size=100.0) == [
        (0.0, 1.0),
        (10.0, 2.0),
    ]


def test_median_is_resistant_to_an_outlier():
    assert median([1.0, 2.0, 100.0]) == 2.0


def test_anomaly_is_flagged_at_the_threshold():
    base = {"median": 10.0, "upper": 20.0}
    policy = {"factor": 1.5}

    assert is_anomaly(30.0, base, policy) is True


def test_baseline_update_respects_the_step_cap():
    base = {"median": 10.0, "upper": 20.0, "freeze_left": 0}
    policy = {"factor": 3.0, "max_step": 2.0, "freeze": 2, "refresh": 1}

    updated = update_baseline(base, 30.0, policy)

    assert updated["median"] == 12.0
    assert updated["median"] <= base["median"] + policy["max_step"]


def test_batch_keeps_entities_independent():
    entities = {
        "low": [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0)],
        "high": [(0.0, 10.0), (1.0, 10.0), (2.0, 10.0), (3.0, 10.0)],
    }
    policy = {
        "window": 100.0,
        "factor": 2.0,
        "max_step": 5.0,
        "freeze": 2,
        "refresh": 1,
    }

    result = anomaly_batch(entities, policy, now=3.0)

    assert result == {"low": False, "high": False}