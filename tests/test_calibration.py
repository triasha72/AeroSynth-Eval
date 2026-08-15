from aerosynth_eval.calibration import (
    CalibrationPoint,
    apply_temperature,
    fit_temperature,
    summarize_calibration,
)


def test_calibration_metrics() -> None:
    points = [
        CalibrationPoint(confidence=0.9, correct=True),
        CalibrationPoint(confidence=0.8, correct=False),
    ]
    summary = summarize_calibration(points)
    assert 0 <= summary.expected_calibration_error <= 1
    assert 0 <= summary.brier_score <= 1
    temperature = fit_temperature(points)
    assert temperature > 0
    assert 0 < apply_temperature(0.8, temperature) < 1
