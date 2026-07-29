"""Root shim — prefer ``python -m harness.check_relative_ranks``."""

from harness.check_relative_ranks import main


if __name__ == "__main__":
    raise SystemExit(main())
