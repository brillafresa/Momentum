"""Backward-compatible wrapper for calibration package entrypoint."""

from calibration.fms_recalib_build_features import *  # noqa: F401,F403
from calibration.fms_recalib_build_features import main


if __name__ == "__main__":
    main()

