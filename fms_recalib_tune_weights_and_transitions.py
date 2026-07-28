"""Backward-compatible wrapper for calibration package entrypoint."""

from calibration.fms_recalib_tune_weights_and_transitions import *  # noqa: F401,F403
from calibration.fms_recalib_tune_weights_and_transitions import main


if __name__ == "__main__":
    main()

