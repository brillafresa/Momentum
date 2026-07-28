"""
Contract: calibration_utils remains a compatibility shim for calibration.session.
"""

from __future__ import annotations


def test_calibration_utils_reexports_session_symbols() -> None:
    import calibration.session as cs
    import calibration_utils as cu

    assert cu.list_sessions is cs.list_sessions
    assert cu.load_session is cs.load_session
    assert cu.save_session is cs.save_session
    assert cu.create_snapshot_id is cs.create_snapshot_id
    assert cu.MergeState is cs.MergeState
    assert cu.SNAPSHOT_ROOT_DIR == cs.SNAPSHOT_ROOT_DIR
    assert cu.SESSION_ROOT_DIR == cs.SESSION_ROOT_DIR

