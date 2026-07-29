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

# Production v4.6.0 — approved zero-based sparse-linear refit.
# Frozen training normalization: median fill → (x - mean) / std → clip[-4, 4].
FMS_FORMULA = (
    "FMS = +0.846427*Z(R2_3M) +0.601307*Z(DD_RECOVERY) "
    "+0.354317*Z(TREND_QUALITY_21D) -0.279017*Z(JUMP_DISCONTINUITY_3M) "
    "-0.196604*Z(UNDER_EMA20_DAYS) +0.186983*Z(R_3M) "
    "-0.181753*Z(STALE_AGE) +0.107915*Z(UP_STREAK_5D) "
    "+0.107766*Z(TREND_EFFICIENCY_REWARD_15D) "
    "-0.104169*Z(RANGE_COMPRESSION_20D)"
)

# 기본 설정
# Batch / scan save threshold used by run_scan_batch (FMS ≥ this value).
# App batch-result viewer may expose its own slider; keep this as the CLI default.
DEFAULT_FMS_THRESHOLD = 0.0
