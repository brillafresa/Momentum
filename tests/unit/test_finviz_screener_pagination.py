"""
Offline validation harness for the Finviz screener parser (no network).

Purpose
-------
Lock the batch universe fetch after Finviz replaced ``table.screener_table`` /
``#pageSelect`` with the ``fv-table`` grid (v5.0.13). The mock HTML is inline
in this file (``_count_text_soup`` = legacy, ``_grid_soup`` = current grid).
It is the fixture. Do not move it into production modules.

Covered
-------
- Legacy ``.count-text`` total and ``#pageSelect`` cap
- Toolbar total when ``.count-text`` is only a filter summary
- Page count ``ceil(N/20)`` without ``pageSelect``
- Grid rows: ``data-ticker`` (not the logo initial), ``Price`` float, ``Change`` fraction
- Partial last-page failure still returns rows already collected
- Legacy HTML still goes through ``overview._parse_table``

Usage (from repo root)
----------------------
    python -m pytest tests/unit/test_finviz_screener_pagination.py -q

FMS scoring is not exercised here. That contract stays in
``tests/unit/test_fms_scoring.py`` and ``tests/unit/test_fms_alive_pullback_production.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from universe_utils import (
    finviz_effective_page_count,
    finviz_screener_view_resilient,
    parse_finviz_total_count,
)


def _count_text_soup(total: int) -> BeautifulSoup:
    html = f'<html><body><div class="count-text">#1 / {total} Total</div></body></html>'
    return BeautifulSoup(html, "lxml")


def test_parse_finviz_total_count_reads_count_text() -> None:
    assert parse_finviz_total_count(_count_text_soup(1090)) == 1090


def _grid_soup(total: int, tickers: list[str]) -> BeautifulSoup:
    """Minimal current Finviz screener grid (no ``pageSelect`` / ``screener_table``)."""
    header = f"""
    <div class="count-text">Filters: sh_price:o10 ta_perf:13wup</div>
    <div class="fv-table_toolbar-count">#1 / {total} Total</div>
    <div class="fv-table_header-cell" data-column-id="row">
      <span class="fv-table_header-label">No.</span>
    </div>
    <div class="fv-table_header-cell" data-column-id="ticker">
      <span class="fv-table_header-label">Ticker</span>
    </div>
    <div class="fv-table_header-cell" data-column-id="company">
      <span class="fv-table_header-label">Company</span>
    </div>
    <div class="fv-table_header-cell" data-column-id="price">
      <span class="fv-table_header-label">Price</span>
    </div>
    <div class="fv-table_header-cell" data-column-id="change">
      <span class="fv-table_header-label">Change %</span>
      <span class="fv-table_header-label-short">Chg %</span>
    </div>
    """
    rows = []
    for i, ticker in enumerate(tickers, start=1):
        rows.append(
            f"""
            <div class="fv-table_row">
              <div class="fv-table_cell" data-column-id="row">{i}</div>
              <div class="fv-table_cell" data-column-id="ticker" data-ticker="{ticker}">
                <span>{ticker[:1]}</span><span>{ticker}</span>
              </div>
              <div class="fv-table_cell" data-column-id="company">{ticker} Inc</div>
              <div class="fv-table_cell" data-column-id="price">172.97</div>
              <div class="fv-table_cell" data-column-id="change">1.50%</div>
            </div>
            """
        )
    return BeautifulSoup(f"<html><body>{header}{''.join(rows)}</body></html>", "lxml")


def test_parse_finviz_total_count_prefers_toolbar_over_filter_summary() -> None:
    """New markup keeps filter codes in ``count-text``; the total lives on the toolbar."""
    assert parse_finviz_total_count(_grid_soup(470, ["A"])) == 470


def test_finviz_effective_page_count_uses_toolbar_total_without_page_select() -> None:
    assert finviz_effective_page_count(_grid_soup(40, ["A"]), page_size=20) == 2


def test_finviz_effective_page_count_caps_page_select_by_total() -> None:
    html = """
    <html><body>
      <div class="count-text">#1 / 1090 Total</div>
      <select id="pageSelect">
        <option>1</option><option>2</option><option>3</option>
      </select>
    </body></html>
    """
    soup = BeautifulSoup(html, "lxml")
    # 1090 rows @ 20/page => 55 pages, even if pageSelect has only 3 options here
    assert finviz_effective_page_count(soup, page_size=20) == 3


def test_finviz_screener_view_resilient_returns_partial_on_last_page_failure() -> None:
    overview = SimpleNamespace(
        size=20,
        request_params={"v": 111},
        reset_calls=0,
    )

    def _reset() -> None:
        overview.reset_calls += 1

    overview.reset = _reset

    first_df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(20)]})
    partial_df = pd.DataFrame({"Ticker": [f"T{i}" for i in range(40)]})

    def _parse_table(df, soup, limit):  # noqa: ANN001
        if df is None:
            return first_df.copy()
        return partial_df.copy()

    overview._parse_table = _parse_table

    soups = [
        _count_text_soup(40),
        _count_text_soup(40),
    ]

    with patch("universe_utils.finviz_effective_page_count", return_value=2), patch(
        "universe_utils.fetch_finviz_screener_page",
        side_effect=[soups[0], RuntimeError("timeout on last page")],
    ):
        out = finviz_screener_view_resilient(overview, sleep_sec=0, allow_partial=True)

    assert len(out) == 20
    assert overview.reset_calls == 1


def test_finviz_screener_view_resilient_parses_fv_grid_ticker_and_pages() -> None:
    """Grid pages keep ``data-ticker`` (not the logo initial) and legacy column names."""
    page1 = [f"T{i:02d}" for i in range(20)]
    page2 = ["A", "AAPL"]
    overview = SimpleNamespace(
        size=20,
        request_params={"v": 111},
        reset_calls=0,
    )
    overview.reset = lambda: setattr(overview, "reset_calls", overview.reset_calls + 1)

    def _parse_table(df, soup, limit):  # noqa: ANN001
        raise AssertionError("legacy screener_table parser must not run on fv-table pages")

    overview._parse_table = _parse_table

    with patch(
        "universe_utils.fetch_finviz_screener_page",
        side_effect=[_grid_soup(22, page1), _grid_soup(22, page2)],
    ):
        out = finviz_screener_view_resilient(overview, sleep_sec=0)

    assert list(out["Ticker"]) == page1 + page2
    assert out.loc[out["Ticker"] == "A", "Company"].iloc[0] == "A Inc"
    assert float(out.loc[out["Ticker"] == "A", "Price"].iloc[0]) == pytest.approx(172.97)
    assert "Change" in out.columns
    assert "Change %" not in out.columns
    assert float(out.loc[out["Ticker"] == "A", "Change"].iloc[0]) == pytest.approx(0.015)
    price_clean = (
        out["Price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    )
    assert (price_clean >= 10.0).all()
    assert overview.reset_calls == 1


def test_finviz_screener_view_resilient_raises_when_partial_disallowed() -> None:
    overview = SimpleNamespace(
        size=20,
        request_params={"v": 111},
        reset_calls=0,
    )
    overview.reset = lambda: setattr(overview, "reset_calls", overview.reset_calls + 1)
    overview._parse_table = lambda df, soup, limit: (
        pd.DataFrame({"Ticker": [f"T{i}" for i in range(20)]}) if df is None
        else pd.DataFrame({"Ticker": [f"T{i}" for i in range(40)]})
    )

    with patch("universe_utils.finviz_effective_page_count", return_value=2), patch(
        "universe_utils.fetch_finviz_screener_page",
        side_effect=[_count_text_soup(40), RuntimeError("timeout on last page")],
    ):
        with pytest.raises(RuntimeError):
            finviz_screener_view_resilient(overview, sleep_sec=0, allow_partial=False)
