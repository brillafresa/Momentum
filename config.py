# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 설정 파일
FMS 전략 및 기타 설정을 관리합니다.
"""

# FMS 전략 정의 (비선형 추세/위치/리스크 결합)
# 요약: FMS = (중·장기 수익률 + R² + EMA50 상대위치 + 조건부 1M 수익률) - (비선형 드로우다운 패널티 + 비선형 20일 변동성 패널티 + 조건부 1M 이벤트성 급등 패널티)
FMS_FORMULA = "비선형 조합: (R_3M, R_6M, R2_3M, EMA50, 조건부 R_1M) - (MaxDD, Vol20, 조건부 R_1M 이벤트성 급등)"

# 기본 설정
DEFAULT_FMS_THRESHOLD = 2.0
