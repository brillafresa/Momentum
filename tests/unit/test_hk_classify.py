from analysis_utils import classify


def test_classify_hk_maps_to_hkg() -> None:
    assert classify("0005.HK") == "HKG"


def test_classify_other_suffixes_unchanged() -> None:
    assert classify("AAPL") == "USA"
    assert classify("005930.KS") == "KOR"
    assert classify("7203.T") == "JPN"

