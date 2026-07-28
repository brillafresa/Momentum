"""
Unit tests for resilient Finviz screener pagination (offline).

Covers total-count parsing, effective page cap, and partial-result fallback when
the final page fetch fails after retries.
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
