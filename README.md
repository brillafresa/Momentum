# KRW Momentum Radar

⚡ **KRW Momentum Radar v2.8**는 다국가 주식 시장의 모멘텀을 실시간으로 분석하고 시각화하는 Streamlit 웹 애플리케이션입니다. 이제 단순한 모니터링 도구를 넘어서 **새로운 투자 기회를 발굴하는 탐색 엔진**으로 진화했습니다.

## 🌟 주요 기능

### 📊 가속 보드

- FMS(Fast Momentum Score) 기반 실시간 모멘텀 분석
- 1일/5일 가속도 변화량 추적
- 상위 N개 종목의 모멘텀 랭킹

### 📈 비교 차트

- 다국가 종목들의 KRW 환산 가격 비교
- 로그/선형 스케일 선택 가능
- 기간별 성과 비교 (3M, 6M, 1Y, 2Y, 5Y)

### 🎯 수익률-변동성 이동맵

- **정적 모드**: 1개월 전 → 어제 → 오늘의 이동 경로 시각화
- **애니메이션 모드**: 최근 10일/20일의 실시간 이동 추적
- 꼬리 효과로 과거 경로 추적 가능

### 📋 상세 분석

- 개별 종목의 EMA(20, 50, 200) 분석
- 최대 낙폭(Drawdown) 추적
- 모멘텀 상태 배지 시스템

### 🚀 신규 종목 탐색 엔진 (NEW!)

- **전체 시장 스캔**: 미국 ETF 시장 전체를 스캔하여 새로운 기회 발굴
- **FMS 기반 랭킹**: 강력한 모멘텀을 가진 신규 종목 자동 발견
- **원클릭 추가**: 발견된 종목을 관심종목에 즉시 추가

### 📋 동적 관심종목 관리 (NEW!)

- **영구 저장**: 관심종목이 세션 종료 후에도 유지
- **실시간 관리**: 직관적인 UI로 관심종목 추가/삭제
- **자동 편출 제안**: 저성과 종목 자동 식별 및 제거 제안

## 🚀 빠른 시작

### 로컬 실행

```bash
# 저장소 클론
git clone <repository-url>
cd Momentum

# 의존성 설치
pip install -r requirements.txt

# 애플리케이션 실행
streamlit run app.py
```

### Streamlit Cloud 배포

1. GitHub에 저장소 푸시
2. [Streamlit Cloud](https://share.streamlit.io)에서 새 앱 생성
3. 저장소 연결 및 `app.py` 파일 지정
4. 자동 배포 완료!

## 📦 포함된 시장

### 🇺🇸 미국 (USD)

- 주요 ETF: SPY, QQQ, VOO, DIA
- 섹터 ETF: XLK, XLF, XLV, SOXX
- 테마 ETF: ICLN, BOTZ, IBB
- 개별주: NVDA, GOOGL

### 🇰🇷 한국 (KRW)

- 대형주: 삼성전자(005930.KS)
- ETF: KODEX, TIGER, ARIRANG 시리즈

### 🇯🇵 일본 (JPY)

- 개별주: 2563.T

## 🔧 설정 옵션

### 차트 기간

- 3M, 6M, 1Y, 2Y, 5Y 선택 가능

### 정렬 기준

- **ΔFMS(1D)**: 1일 가속도 변화
- **ΔFMS(5D)**: 5일 가속도 변화
- **FMS(현재)**: 현재 모멘텀 점수
- **1M 수익률**: 1개월 수익률

### 표시 옵션

- Top N: 5~60개 종목 선택
- 로그 스케일: 비교 차트용
- 꼬리 길이: 0~10일 이동 경로

## 📊 지표 설명

### FMS (Fast Momentum Score)

```
FMS = 0.5×Z(1M수익률) + 0.3×Z(30일기울기) + 0.2×Z(EMA50상대위치)
      + 0.1×Z(120일돌파) - 0.1×Z(20일변동성)
```

### 주요 지표

- **R_1M**: 1개월 수익률
- **AboveEMA50**: EMA50 대비 현재가 위치
- **Breakout120**: 120일 최고가 대비 위치
- **Slope30**: 30일 로그 기울기 (연율화)
- **Vol20**: 20일 변동성 (연율화)

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Data**: yfinance (Yahoo Finance API)
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Timezone**: pytz (KST 기준)

## 📝 버전 히스토리

### v2.8 (현재)

- **관심종목 영구 저장**: CSV 파일을 통한 관심종목 영구 저장
- **신규 종목 탐색 엔진**: 전체 미국 ETF 시장 스캔 기능
- **진부한 종목 자동 편출**: FMS 기반 저성과 종목 제거 제안
- **동적 포트폴리오 관리**: 실시간 관심종목 추가/삭제
- **모듈화**: 관심종목 관리 로직을 별도 모듈로 분리

### v2.7

- FMS 설명 박스 추가로 사용자 이해도 향상
- 애니메이션 자동 재생 및 꼬리 길이 자동 조정
- 시각적 개선 (과거 시점과 꼬리 중복 제거)
- UI 간소화 (상단 KPI 제거, 기본값 최적화)

### v2.6

- FMS 기반 모멘텀 스코어링
- 다국가 시장 통합 분석
- KRW 환산 가격 비교

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## ⚠️ 면책 조항

이 도구는 교육 및 연구 목적으로만 제공됩니다. 투자 결정에 사용하기 전에 반드시 전문가의 조언을 구하시기 바랍니다. 과거 성과가 미래 결과를 보장하지 않습니다.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해 주세요.

---

**⚡ KRW Momentum Radar** - 모멘텀의 힘을 시각화하세요!
