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

# Production v5.0.0 — alive_pullback absolute nonlinear (SEG_* + residual features).
# Locked params: core.fms_features.PRODUCTION_ALIVE_PULLBACK_PARAMS
# (session cal_fms_20260730_190637 / round-2b MC). No watchlist-relative Z.
FMS_FORMULA = (
    "FMS = softplus(R_3M - floor/2) * ("
    "w_recent*sgn(SEG_RET_0_5)*|SEG_RET_0_5|^pow + alive_boost*softplus(SEG_RET_0_5)*PRIOR_SUPPORT "
    "+ w_mid_pos*softplus(SEG_RET_5_21) + w_mid_neg_forgive*MID_DIP_RECOVERY*(0.5+PRIOR_SUPPORT) "
    "+ w_prior*SEG_RET_21_63*(0.5+PRIOR_SUPPORT) + w_abs*softplus(R_3M-floor) "
    "+ w_breadth*(RECENT_UP_DAYS_5D/5) + w_grind*grind + w_eff*softplus(TREND_EFFICIENCY_REWARD_15D)"
    ") / (1 + w_stale_run*STALE_AFTER_RUN + w_jump_share*softplus(RECENT_JUMP_SHARE_5D-0.55)); "
    "absolute path score (no watchlist Z); tradeability may force FMS=-999"
)

# 기본 설정
# Batch / scan save threshold used by run_scan_batch.
# v5.0.0 scores are absolute; softplus floor keeps weak absolute-return paths near 0
# while disqualified names remain -999. Keep CLI default at 0.0.
DEFAULT_FMS_THRESHOLD = 0.0
