from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransportRunSummary:
    mode: str
    total: int
    passed: int
    failed: int
    accuracy: float
    execution_error_count: int = 0


def _assert_same_shape(runs: list[TransportRunSummary]) -> None:
    totals = {run.total for run in runs}
    assert len(totals) == 1, f"Transport modes disagree on total case count: {totals}"


def _assert_accuracy_within_tolerance(
    runs: list[TransportRunSummary], tolerance: float
) -> None:
    accuracies = [run.accuracy for run in runs]
    assert max(accuracies) - min(accuracies) <= tolerance, (
        f"Transport accuracy drift too large: {accuracies}, tolerance={tolerance}"
    )


def _assert_no_execution_errors(runs: list[TransportRunSummary]) -> None:
    offenders = {
        run.mode: run.execution_error_count
        for run in runs
        if run.execution_error_count > 0
    }
    assert not offenders, f"Transport run contains execution errors: {offenders}"


def test_transport_consistency_accepts_small_expected_drift() -> None:
    runs = [
        TransportRunSummary(mode="base64", total=40, passed=24, failed=16, accuracy=0.6000),
        TransportRunSummary(mode="local_path", total=40, passed=24, failed=16, accuracy=0.6000),
        TransportRunSummary(mode="http", total=40, passed=24, failed=16, accuracy=0.6000),
    ]
    _assert_same_shape(runs)
    _assert_accuracy_within_tolerance(runs, tolerance=0.02)
    _assert_no_execution_errors(runs)


def test_transport_consistency_flags_large_drift() -> None:
    runs = [
        TransportRunSummary(mode="base64", total=10, passed=10, failed=0, accuracy=1.0),
        TransportRunSummary(mode="local_path", total=10, passed=10, failed=0, accuracy=1.0),
        TransportRunSummary(mode="http", total=10, passed=6, failed=4, accuracy=0.6),
    ]
    try:
        _assert_accuracy_within_tolerance(runs, tolerance=0.1)
    except AssertionError as exc:
        assert "Transport accuracy drift too large" in str(exc)
    else:
        raise AssertionError("Expected large transport drift to be rejected")


def test_transport_consistency_flags_case_count_mismatch() -> None:
    runs = [
        TransportRunSummary(mode="base64", total=40, passed=24, failed=16, accuracy=0.6),
        TransportRunSummary(mode="local_path", total=40, passed=24, failed=16, accuracy=0.6),
        TransportRunSummary(mode="http", total=39, passed=24, failed=15, accuracy=0.6154),
    ]
    try:
        _assert_same_shape(runs)
    except AssertionError as exc:
        assert "Transport modes disagree on total case count" in str(exc)
    else:
        raise AssertionError("Expected transport case-count mismatch to be rejected")


def test_transport_consistency_flags_execution_errors() -> None:
    runs = [
        TransportRunSummary(
            mode="base64",
            total=10,
            passed=0,
            failed=10,
            accuracy=0.0,
            execution_error_count=10,
        ),
        TransportRunSummary(
            mode="local_path",
            total=10,
            passed=0,
            failed=10,
            accuracy=0.0,
            execution_error_count=10,
        ),
        TransportRunSummary(
            mode="http",
            total=10,
            passed=0,
            failed=10,
            accuracy=0.0,
            execution_error_count=10,
        ),
    ]
    try:
        _assert_no_execution_errors(runs)
    except AssertionError as exc:
        assert "Transport run contains execution errors" in str(exc)
    else:
        raise AssertionError("Expected execution-error transport run to be rejected")
