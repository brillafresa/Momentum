"""Unit tests for calibration.session helpers."""

from __future__ import annotations

import json
import os

from calibration.session import latest_completed_session, list_sessions, save_session, session_path


def test_list_sessions_excludes_baseline_metrics_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("calibration.session.SESSION_ROOT_DIR", str(tmp_path))

    save_session("cal_fms_20260729_120000", {"snapshot_id": "fms_20260729_120000", "phase": "done"})

    baseline_path = tmp_path / "cal_fms_20260729_120000__baseline_metrics.json"
    baseline_path.write_text(
        json.dumps({"session_id": "cal_fms_20260729_120000", "metrics": {}}),
        encoding="utf-8",
    )

    sessions = list_sessions()
    assert sessions == ["cal_fms_20260729_120000"]
    assert not any("__baseline_metrics" in sid for sid in sessions)


def test_list_sessions_orders_by_saved_at_not_filename(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("calibration.session.SESSION_ROOT_DIR", str(tmp_path))

    old_path = session_path("cal_fms_20260311_161339")
    new_path = session_path("20260729_recent1m")
    tmp_path.mkdir(parents=True, exist_ok=True)
    with open(old_path, "w", encoding="utf-8") as f:
        json.dump({"saved_at": "2026-07-29T19:45:42", "phase": "done"}, f)
    with open(new_path, "w", encoding="utf-8") as f:
        json.dump({"saved_at": "2026-07-29T20:00:00", "phase": "done"}, f)
    # Make filesystem mtime contradict saved_at ordering.
    os.utime(old_path, (3_000, 3_000))
    os.utime(new_path, (1_000, 1_000))

    assert list_sessions() == ["20260729_recent1m", "cal_fms_20260311_161339"]


def test_list_sessions_skips_missing_saved_at(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("calibration.session.SESSION_ROOT_DIR", str(tmp_path))

    valid_path = session_path("valid_session")
    invalid_path = session_path("legacy_no_saved_at")
    tmp_path.mkdir(parents=True, exist_ok=True)
    with open(valid_path, "w", encoding="utf-8") as f:
        json.dump({"saved_at": "2026-07-29T20:00:00", "phase": "done"}, f)
    with open(invalid_path, "w", encoding="utf-8") as f:
        json.dump({"phase": "done"}, f)
    os.utime(invalid_path, (9_000, 9_000))

    assert list_sessions() == ["valid_session"]


def test_latest_completed_session_ignores_newer_incomplete_session(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("calibration.session.SESSION_ROOT_DIR", str(tmp_path))

    done_path = save_session(
        "latest_done",
        {"phase": "done", "final_ranking": ["A", "B"]},
    )
    review_path = save_session(
        "newer_review",
        {"phase": "review", "final_ranking": ["B", "A"]},
    )
    os.utime(done_path, (1_000, 1_000))
    os.utime(review_path, (2_000, 2_000))

    session_id, session = latest_completed_session()
    assert session_id == "latest_done"
    assert session["final_ranking"] == ["A", "B"]
