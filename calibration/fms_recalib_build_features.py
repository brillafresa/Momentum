"""
FMS 재보정을 위한 피처 테이블 생성 스크립트.

- 입력: 최신 완료 세션 manifest + 고정 스냅샷
- 출력: fms_recalib_features.csv, fms_recalib_manifest.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from calibration.manifest import MANIFEST_PATH, assert_manifest_fresh, build_manifest, load_manifest
from calibration.ranking_metrics import compute_metrics
from calibration.session import SNAPSHOT_ROOT_DIR
from core.fms import score_fms_from_feature_frame
from core.fms_features import build_panel_feature_frame

OUT_PATH = "fms_recalib_features.csv"


def main() -> None:
    manifest = build_manifest()
    snapshot_id = manifest.snapshot_id
    ranking = manifest.ranking
    snap_path = os.path.join(SNAPSHOT_ROOT_DIR, snapshot_id, "prices_krw.pkl")
    if not os.path.exists(snap_path):
        print(f"스냅샷이 없습니다: {snap_path}")
        return

    prices_krw = pd.read_pickle(snap_path)
    assert_manifest_fresh(manifest, ranking=ranking, prices=prices_krw)

    cols = [c for c in ranking if c in prices_krw.columns]
    if len(cols) != len(ranking):
        missing = [c for c in ranking if c not in prices_krw.columns]
        raise RuntimeError(f"ranking symbols missing from snapshot: {missing[:5]}")

    features = build_panel_feature_frame(prices_krw, symbols=cols)
    features = features.loc[cols]
    features["rank"] = range(1, len(features) + 1)
    features.to_csv(OUT_PATH, encoding="utf-8-sig")
    print("Wrote", OUT_PATH, "shape", features.shape)
    print("Wrote", MANIFEST_PATH)

    baseline = score_fms_from_feature_frame(features)
    metrics = compute_metrics(features["rank"], baseline)
    baseline_payload = {
        "session_id": manifest.session_id,
        "snapshot_id": manifest.snapshot_id,
        "manifest_path": MANIFEST_PATH,
        "features_csv": OUT_PATH,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_symbols": int(features.shape[0]),
        "metrics": {
            "inversion_rate": metrics.inv,
            "spearman_rho": metrics.rho,
            "pair_delta_error": metrics.pair_err,
        },
    }
    out_json = os.path.join(
        "fms_calibration_sessions", f"{manifest.session_id}__baseline_metrics.json"
    )
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(baseline_payload, f, ensure_ascii=False, indent=2)
    print("Wrote", out_json)


if __name__ == "__main__":
    main()
