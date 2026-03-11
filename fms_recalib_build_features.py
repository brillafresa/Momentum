"""
FMS 재보정을 위한 피처 테이블 생성 스크립트.

- 입력: 최신 세션(fms_calibration_sessions/) 및 해당 스냅샷(fms_calibration_snapshots/)
- 출력: fms_recalib_features.csv (R_1M, R_3M, R_6M, R2_3M, AboveEMA50, Vol20_Ann, MaxDD_Pct, rank)

실행 전 UI에서 FMS 재보정 A/B 비교를 완료해 세션을 저장해 두어야 합니다.
"""

import os
import numpy as np
import pandas as pd

from analysis_utils import r_squared_3m, returns_pct, last_vol_annualized, ema
from calibration_utils import list_sessions, load_session, SNAPSHOT_ROOT_DIR

OUT_PATH = "fms_recalib_features.csv"


def main() -> None:
    sessions = list_sessions()
    if not sessions:
        print("세션이 없습니다. UI에서 FMS 재보정 A/B 비교를 완료해 세션을 저장한 뒤 다시 실행하세요.")
        return
    session_id = sessions[0]
    session = load_session(session_id)
    ranking = session.get("final_ranking") or []
    snapshot_id = session.get("snapshot_id") or session_id.replace("cal_", "")
    snap_path = os.path.join(SNAPSHOT_ROOT_DIR, snapshot_id, "prices_krw.pkl")
    if not os.path.exists(snap_path):
        print(f"스냅샷이 없습니다: {snap_path}")
        return

    prices_krw = pd.read_pickle(snap_path)
    # 관심 종목 순서대로 정렬, 누락 열은 건너뜀
    cols = [c for c in ranking if c in prices_krw.columns]
    prices_krw = prices_krw[cols].dropna(how="all")

    # 기본 수익률/지표
    r_1m = returns_pct(prices_krw, 21).rename("R_1M")
    r_3m = returns_pct(prices_krw, 63).rename("R_3M")
    r_6m = returns_pct(prices_krw, 126).rename("R_6M")
    R2_3m = r_squared_3m(prices_krw).rename("R2_3M")
    vol20 = last_vol_annualized(prices_krw, 20).rename("Vol20_Ann")

    # EMA50 상대 위치
    above_ema50 = {}
    for c in prices_krw.columns:
        s = prices_krw[c].dropna()
        if s.empty:
            above_ema50[c] = np.nan
            continue
        e50 = ema(s, 50)
        if e50.iloc[-1] > 0:
            above_ema50[c] = s.iloc[-1] / e50.iloc[-1] - 1.0
        else:
            above_ema50[c] = np.nan
    above_ema50 = pd.Series(above_ema50, name="AboveEMA50")

    # 최대 드로우다운(%)
    md_dict = {}
    for c in prices_krw.columns:
        s = prices_krw[c].dropna()
        if s.empty:
            md_dict[c] = np.nan
            continue
        roll_max = s.cummax()
        dd = (s / roll_max - 1.0) * 100.0
        md_dict[c] = float(dd.min())
    max_dd = pd.Series(md_dict, name="MaxDD_Pct")

    features = pd.concat(
        [r_1m, r_3m, r_6m, R2_3m, above_ema50, vol20, max_dd], axis=1
    )
    # 정답 순서대로 재정렬하고 rank 부여
    features = features.loc[[c for c in ranking if c in features.index]]
    features["rank"] = range(1, len(features) + 1)

    features.to_csv(OUT_PATH, encoding="utf-8-sig")
    print("Wrote", OUT_PATH, "shape", features.shape)


if __name__ == "__main__":
    main()

