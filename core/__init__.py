"""Pure business logic package (no network I/O).

Migrated so far:
- ``core.indicators`` — ``ema``, ``returns_pct``, ``r_squared_3m``, ``ytd_return``, ``last_vol_annualized``
- ``core.tradeability`` — ``calculate_tradeability_filters``
- ``core.fms`` — ``compute_fms_snapshot``, ``momentum_now_and_delta``, ``_mom_snapshot``,
  ``score_fms_from_feature_frame``

Still in ``analysis_utils`` (transitional): downloads, filter debug, batch orchestration.

Boundary rules (see HARNESS_RULES.md):
- Do **not** import yfinance, finvizfinance, requests, or streamlit here.
- Do **not** re-export from ``analysis_utils`` in this package init — that would
  pull the network stack into an ostensibly pure package.
- Callers may keep importing via ``analysis_utils`` (re-export shim)
  or directly from ``core.*``.
"""

__all__: list[str] = []
