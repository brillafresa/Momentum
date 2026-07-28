from unittest.mock import patch

import pandas as pd
import pandas.testing as pdt

import analysis_utils


def test_download_fx_computes_hkdkrw_via_hkdusd() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="B")

    usdkrw_expected = pd.Series([1200, 1210, 1190, 1220, 1230, 1240], index=idx, name="USDKRW")
    usdjpy_expected = pd.Series([110.0, 111.0, 109.5, 112.0, 113.0, 114.0], index=idx, name="USDJPY")
    # HKDUSD: HKD per USD (yfinance ticker HKD=X convention)
    hkdusd_expected = pd.Series([7.75, 7.76, 7.74, 7.77, 7.78, 7.79], index=idx, name="HKDUSD")

    hkdkrw_expected = (usdkrw_expected / hkdusd_expected).rename("HKDKRW")

    def _fake_download_prices(tickers, period_, interval, **kwargs):
        t = list(tickers)[0]
        if t == "KRW=X":
            return pd.DataFrame({"KRW=X": usdkrw_expected}, index=idx), []
        if t == "JPY=X":
            return pd.DataFrame({"JPY=X": usdjpy_expected}, index=idx), []
        if t == "HKD=X":
            return pd.DataFrame({"HKD=X": hkdusd_expected}, index=idx), []
        return pd.DataFrame(), [t]

    with patch("analysis_utils.download_prices", side_effect=_fake_download_prices):
        usdkrw, usdjpy, jpykrw, hkdkrw = analysis_utils.download_fx(period_="1y", interval="1d")

    pdt.assert_series_equal(usdkrw, usdkrw_expected)
    pdt.assert_series_equal(usdjpy, usdjpy_expected)
    pdt.assert_series_equal(hkdkrw, hkdkrw_expected)

    # Sanity check: JPYKRW = USDKRW / USDJPY
    pdt.assert_series_equal(jpykrw, (usdkrw_expected / usdjpy_expected).rename("JPYKRW"))

