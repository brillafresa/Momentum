# -*- coding: utf-8 -*-
"""
KRW Momentum Radar - 설정 파일
FMS 전략 및 기타 설정을 관리합니다.
"""

# FMS 전략 정의
FMS_STRATEGIES = {
    'Standard': {
        'formula': "0.5 * Z('1M수익률') + 0.3 * Z('30일기울기') + 0.2 * Z('EMA50상대위치') + 0.1 * Z('120일돌파') - 0.1 * Z('20일변동성')",
        'description': "기본 FMS 전략입니다. 1개월 수익률에 높은 가중치를 부여하여 단기 모멘텀을 중시합니다."
    },
    'Stable Growth': {
        'formula': "0.4 * Z('1M수익률') + 0.3 * Z('3M수익률') + 0.2 * Z('EMA50상대위치') - 0.4 * Z('20일변동성') - 0.4 * Z('변동성 가속도')",
        'description': "추세의 지속성과 안정성을 중시합니다. 단기 변동성 및 변동성의 급격한 증가에 강력한 페널티를 부여하여, 꾸준히 우상향하는 종목을 탐색합니다."
    }
}

# 기본 설정
DEFAULT_FMS_STRATEGY = 'Stable Growth'
DEFAULT_FMS_THRESHOLD = 2.0
