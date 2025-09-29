# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 설정 파일
FMS 전략 및 기타 설정을 관리합니다.
"""

# FMS 전략 정의 (Stable Growth 단일화)
FMS_FORMULA = "0.4 * Z('1M수익률') + 0.3 * Z('3M수익률') + 0.2 * Z('EMA50상대위치') - 0.4 * Z('20일변동성')"

# 기본 설정
DEFAULT_FMS_THRESHOLD = 2.0
