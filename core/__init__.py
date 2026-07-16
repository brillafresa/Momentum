"""Pure business logic package (no network I/O).

Target home for FMS / indicators / tradeability after gradual migration
from ``analysis_utils.py``.

Boundary rules (see HARNESS_RULES.md):
- Do **not** import yfinance, finvizfinance, requests, or streamlit here.
- Do **not** re-export from ``analysis_utils`` in this package init — that would
  pull the network stack into an ostensibly pure package.
- Until modules are migrated, production and tests import scoring from
  ``analysis_utils`` (``compute_fms_snapshot``, ``momentum_now_and_delta``).
"""

__all__: list[str] = []
