"""
FMS 재보정을 위한 피처 테이블 생성 스크립트.

- 입력: 최신 세션(fms_calibration_sessions/) 및 해당 스냅샷(fms_calibration_snapshots/)
- 출력: fms_recalib_features.csv (R_1M, R_3M, R_6M, R2_3M, AboveEMA50, Vol20_Ann, MaxDD_Pct, rank)

실행 전 UI에서 FMS 재보정 A/B 비교를 완료해 세션을 저장해 두어야 합니다.
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd

from analysis_utils import r_squared_3m, returns_pct, last_vol_annualized, ema
from calibration_utils import list_sessions, load_session, SNAPSHOT_ROOT_DIR, SESSION_ROOT_DIR

OUT_PATH = "fms_recalib_features.csv"


def main() -> None:
    sessions = list_sessions()
    if not sessions:
        print("세션이 없습니다. UI에서 FMS 재보정 A/B 비교를 완료해 세션을 저장한 뒤 다시 실행하세요.")
        return

    # 가장 최신 세션들 중에서 final_ranking 이 비어 있지 않은 세션을 찾습니다.
    session_id = None
    session = None
    ranking: list[str] = []
    for sid in sessions:
        s = load_session(sid)
        r = s.get("final_ranking") or []
        if r:
            session_id = sid
            session = s
            ranking = r
            break

    if session_id is None or not ranking:
        print("final_ranking 이 있는 세션을 찾지 못했습니다. UI에서 정렬을 완료한 뒤 다시 실행하세요.")
        return
    snapshot_id = session.get("snapshot_id") or session_id.replace("cal_", "")
    snap_path = os.path.join(SNAPSHOT_ROOT_DIR, snapshot_id, "prices_krw.pkl")
    if not os.path.exists(snap_path):
        print(f"스냅샷이 없습니다: {snap_path}")
        return

    prices_krw = pd.read_pickle(snap_path)
    # 관심 종목 순서대로 정렬, 누락 열은 건너뜀
    cols = [c for c in ranking if c in prices_krw.columns]
    prices_krw = prices_krw[cols].dropna(how="all")

    # 기본 수익률/지표 (기존 7개)
    r_1m = returns_pct(prices_krw, 21).rename("R_1M")
    r_3m = returns_pct(prices_krw, 63).rename("R_3M")
    r_6m = returns_pct(prices_krw, 126).rename("R_6M")
    R2_3m = r_squared_3m(prices_krw).rename("R2_3M")
    vol20 = last_vol_annualized(prices_krw, 20).rename("Vol20_Ann")

    # 추가 수익률(단기/초단기)
    r_10d = returns_pct(prices_krw, 10).rename("R_10D")
    r_5d = returns_pct(prices_krw, 5).rename("R_5D")

    # EMA50 상대 위치 + EMA20 관련 파생 변수들
    above_ema50 = {}
    ema20_slope_10d = {}
    ema20_curv_20d = {}
    under_ema20_depth = {}
    under_ema20_days = {}
    down_streak_5d = {}

    for c in prices_krw.columns:
        s = prices_krw[c].dropna()
        if s.empty:
            above_ema50[c] = np.nan
            ema20_slope_10d[c] = np.nan
            ema20_curv_20d[c] = np.nan
            under_ema20_depth[c] = np.nan
            under_ema20_days[c] = np.nan
            down_streak_5d[c] = np.nan
            continue

        e50 = ema(s, 50)
        if e50.iloc[-1] > 0:
            above_ema50[c] = s.iloc[-1] / e50.iloc[-1] - 1.0
        else:
            above_ema50[c] = np.nan

        # EMA20 기반 파생 변수들
        e20 = ema(s, 20)

        # 최근 10일 EMA20 기울기 (로그 스케일에서의 선형 회귀 기울기)
        if len(e20) >= 10:
            last10 = e20.iloc[-10:]
            x10 = np.arange(len(last10), dtype=float)
            y10 = np.log(last10.replace(0, np.nan)).dropna()
            if len(y10) == len(x10):
                coef = np.polyfit(x10, y10, 1)[0]
                ema20_slope_10d[c] = float(coef)
            else:
                ema20_slope_10d[c] = np.nan
        else:
            ema20_slope_10d[c] = np.nan

        # 최근 20일 EMA20 곡률 근사 (앞/뒤 10일 기울기 차이)
        if len(e20) >= 20:
            first10 = e20.iloc[-20:-10]
            last10 = e20.iloc[-10:]
            x_seg = np.arange(10, dtype=float)
            def _slope(seg: pd.Series) -> float:
                y = np.log(seg.replace(0, np.nan)).dropna()
                if len(y) != len(x_seg):
                    return np.nan
                return float(np.polyfit(x_seg, y, 1)[0])
            s_first = _slope(first10)
            s_last = _slope(last10)
            if np.isnan(s_first) or np.isnan(s_last):
                ema20_curv_20d[c] = np.nan
            else:
                ema20_curv_20d[c] = s_last - s_first
        else:
            ema20_curv_20d[c] = np.nan

        # 최근 60일 EMA20 아래 이탈 깊이/일수
        tail60 = s.iloc[-60:] if len(s) >= 60 else s
        e20_60 = e20.reindex(tail60.index)
        mask_under = tail60 < e20_60
        if not mask_under.any():
            under_ema20_depth[c] = 0.0
            under_ema20_days[c] = 0
        else:
            rel = tail60[mask_under] / e20_60[mask_under] - 1.0
            under_ema20_depth[c] = float(rel.min())
            under_ema20_days[c] = int(mask_under.sum())

        # 최근 5일 연속 하락 최대 길이
        if len(s) >= 5:
            last5 = s.iloc[-5:]
            diff = last5.diff()
            is_down = diff < 0
            max_run = 0
            cur = 0
            # 첫 번째 값은 NaN이므로 두 번째부터 검사
            for v in is_down.iloc[1:]:
                if bool(v):
                    cur += 1
                    max_run = max(max_run, cur)
                else:
                    cur = 0
            down_streak_5d[c] = int(max_run)
        else:
            down_streak_5d[c] = np.nan

    above_ema50 = pd.Series(above_ema50, name="AboveEMA50")
    ema20_slope_10d = pd.Series(ema20_slope_10d, name="EMA20_SLOPE_10D")
    ema20_curv_20d = pd.Series(ema20_curv_20d, name="EMA20_CURV_20D")
    under_ema20_depth = pd.Series(under_ema20_depth, name="UNDER_EMA20_DEPTH")
    under_ema20_days = pd.Series(under_ema20_days, name="UNDER_EMA20_DAYS")
    down_streak_5d = pd.Series(down_streak_5d, name="DOWN_STREAK_5D")

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
        [
            r_1m,
            r_3m,
            r_6m,
            R2_3m,
            above_ema50,
            vol20,
            max_dd,
            r_10d,
            r_5d,
            ema20_slope_10d,
            ema20_curv_20d,
            under_ema20_depth,
            under_ema20_days,
            down_streak_5d,
        ],
        axis=1,
    )
    # 정답 순서대로 재정렬하고 rank 부여
    features = features.loc[[c for c in ranking if c in features.index]]
    features["rank"] = range(1, len(features) + 1)

    features.to_csv(OUT_PATH, encoding="utf-8-sig")
    print("Wrote", OUT_PATH, "shape", features.shape)

    # --- Baseline metrics snapshot (per-session) ---
    # 새 정답셋을 만들 때마다, 해당 정답셋 기준으로 current FMS의 baseline 지표/순위를 저장합니다.
    # 정답셋이 바뀌면 지표 절대값은 당연히 달라지므로, "세션 내부"에서 current vs proposed만 비교하면 됩니다.
    try:
        from fms_recalib_evaluate_formulas import f_current, pairwise_inversion_rate
        from fms_recalib_rank_metrics import compute_pairwise_rank_delta_error, score_to_model_rank
        from scipy.stats import spearmanr

        true_rank = features["rank"]
        score = f_current(features)
        inv = float(pairwise_inversion_rate(true_rank, score))
        model_rank = score_to_model_rank(score)
        common = pd.concat([true_rank, model_rank], axis=1).dropna()
        rho, _ = spearmanr(common.iloc[:, 0], common.iloc[:, 1])
        rho = float(rho)
        pair_err = float(compute_pairwise_rank_delta_error(true_rank, model_rank))

        baseline = {
            "session_id": session_id,
            "snapshot_id": snapshot_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "features_csv": OUT_PATH,
            "n_symbols": int(features.shape[0]),
            "metrics": {
                "inversion_rate": inv,
                "spearman_rho": rho,
                "pair_delta_error": pair_err,
            },
            # 추후 비교/디버깅을 위해 저장 (정답셋 내부에서만 의미 있음)
            "ranks": {
                sym: {
                    "true_rank": int(true_rank.loc[sym]) if sym in true_rank.index else None,
                    "model_rank": int(model_rank.loc[sym]) if sym in model_rank.index and not pd.isna(model_rank.loc[sym]) else None,
                    "score": float(score.loc[sym]) if sym in score.index and not pd.isna(score.loc[sym]) else None,
                }
                for sym in features.index.astype(str).tolist()
            },
        }

        os.makedirs(SESSION_ROOT_DIR, exist_ok=True)
        out_json = os.path.join(SESSION_ROOT_DIR, f"{session_id}__baseline_metrics.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(baseline, f, ensure_ascii=False, indent=2)
        print("Wrote", out_json)
    except Exception as e:
        print(f"[warn] baseline_metrics 저장 실패: {e}")


if __name__ == "__main__":
    main()

