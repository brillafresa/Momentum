"""Contracts for short-horizon candidate screening metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from calibration.fms_recalib_screen_short_horizon import _compute_metrics_fast
from calibration.fms_recalib_tune_weights_and_transitions import compute_metrics


def test_vectorized_metrics_match_existing_recalibration_metrics() -> None:
    frame = pd.DataFrame(
        {"rank": [1, 2, 3, 4, 5]},
        index=["A", "B", "C", "D", "E"],
    )
    score = pd.Series(
        {"A": 0.8, "B": 1.1, "C": 0.4, "D": -0.2, "E": 0.1},
        name="FMS",
    )

    expected = compute_metrics(frame, score)
    actual = _compute_metrics_fast(frame["rank"], score)

    assert actual.inv == pytest.approx(expected.inv)
    assert actual.rho == pytest.approx(expected.rho)
    assert actual.pair_err == pytest.approx(expected.pair_err)
