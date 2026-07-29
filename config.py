# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 설정 파일
FMS 전략 및 기타 설정을 관리합니다.

Production note
---------------
이 모듈은 운영 UI/배치가 참조하는 **런타임 설정**만 둔다.
테스트용 Mock 경로·fixture 파일명·하네스 플래그는 여기 넣지 않는다.
검증용 설정은 ``tests/`` · ``harness/`` · ``scripts/`` 에 둔다.
"""

# Production v4.7.0 — current-watchlist relative Z + cash-like bonus gate.
# Each axis: reference median fill → (x - watchlist mean) / watchlist std → ±4.
# Cash-like gate: when low R_3M ∧ ultra-low Vol20_Ann ∧ high R2_3M, suppress
# positive contributions on quality axes other than R_3M (penalties unchanged).
FMS_FORMULA = (
    "FMS = +0.846427*Z(R2_3M) +0.601307*Z(DD_RECOVERY) "
    "+0.354317*Z(TREND_QUALITY_21D) -0.279017*Z(JUMP_DISCONTINUITY_3M) "
    "-0.196604*Z(UNDER_EMA20_DAYS) +0.186983*Z(R_3M) "
    "-0.181753*Z(STALE_AGE) +0.107915*Z(UP_STREAK_5D) "
    "+0.107766*Z(TREND_EFFICIENCY_REWARD_15D) "
    "-0.104169*Z(RANGE_COMPRESSION_20D); "
    "Z: current account watchlist mean/std; "
    "cash_like_gate: positive quality bonuses × (1 - "
    "low_return(R_3M)×ultra_low_vol(Vol20_Ann)×high_smooth(R2_3M))"
)

# 기본 설정
# Batch / scan save threshold used by run_scan_batch. In v4.7.0, FMS ≥ 0
# means the candidate is at/above the current account-watchlist relative baseline.
# App batch-result viewer may expose its own slider; keep this as the CLI default.
DEFAULT_FMS_THRESHOLD = 0.0
