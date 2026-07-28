import pandas as pd

import universe_utils


def test_load_universe_free_merges_hk_only_when_file_exists(monkeypatch) -> None:
    def _exists(path: str) -> bool:
        return path in {
            "screened_universe.csv",
            "korean_universe.csv",
            "hongkong_universe.csv",
        }

    def _read_csv(path: str, *args, **kwargs) -> pd.DataFrame:
        if path == "screened_universe.csv":
            return pd.DataFrame({"Symbol": ["AAPL"]})
        if path == "korean_universe.csv":
            return pd.DataFrame({"Symbol": ["005930.KS"]})
        if path == "hongkong_universe.csv":
            return pd.DataFrame({"Symbol": ["0005.HK"]})
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(universe_utils.os.path, "exists", _exists)
    monkeypatch.setattr(universe_utils.pd, "read_csv", _read_csv)

    ok, symbols, _msg = universe_utils.load_universe_file(mode=universe_utils.MODE_FREE)
    assert ok is True
    assert symbols == ["AAPL", "005930.KS", "0005.HK"]


def test_load_universe_irp_does_not_include_hk(monkeypatch) -> None:
    def _exists(path: str) -> bool:
        return path in {"korean_etf_univers.csv"}

    def _read_csv(path: str, *args, **kwargs) -> pd.DataFrame:
        if path == "korean_etf_univers.csv":
            return pd.DataFrame({"Symbol": ["005930.KS"]})
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr(universe_utils.os.path, "exists", _exists)
    monkeypatch.setattr(universe_utils.pd, "read_csv", _read_csv)

    ok, symbols, _msg = universe_utils.load_universe_file(mode=universe_utils.MODE_IRP)
    assert ok is True
    assert symbols == ["005930.KS"]

