"""Backward-compatible wrapper for calibration package entrypoint."""

from calibration.fms_recalib_inspect_patterns import *  # noqa: F401,F403
from calibration.fms_recalib_inspect_patterns import main


if __name__ == "__main__":
    main()

