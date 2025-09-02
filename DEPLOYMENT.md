# 배포 가이드

이 문서는 KRW Momentum Radar를 다양한 플랫폼에 배포하는 방법을 안내합니다.

## 🚀 Streamlit Cloud 배포

### 1. GitHub 저장소 준비

1. **저장소 생성**: GitHub에 새로운 저장소를 생성합니다.
2. **파일 업로드**: 다음 파일들이 포함되어야 합니다:
   - `app.py` (메인 애플리케이션)
   - `requirements.txt` (의존성)
   - `README.md` (프로젝트 설명)
   - `.streamlit/config.toml` (Streamlit 설정)

### 2. Streamlit Cloud 설정

1. **Streamlit Cloud 접속**: [share.streamlit.io](https://share.streamlit.io) 방문
2. **GitHub 로그인**: GitHub 계정으로 로그인
3. **새 앱 생성**: "New app" 버튼 클릭
4. **저장소 선택**: 생성한 GitHub 저장소 선택
5. **설정 구성**:
   - **Main file path**: `app.py`
   - **App URL**: 원하는 URL 설정 (선택사항)
   - **Python version**: 3.9+ 권장

### 3. 배포 실행

1. **Deploy**: "Deploy!" 버튼 클릭
2. **빌드 대기**: 초기 빌드에는 5-10분 소요
3. **배포 완료**: 성공 시 앱 URL 제공

### 4. 배포 후 확인

- [ ] 앱이 정상적으로 로드되는지 확인
- [ ] 모든 차트가 올바르게 표시되는지 확인
- [ ] 데이터 업데이트가 정상 작동하는지 확인
- [ ] 에러 로그 확인 (Streamlit Cloud 대시보드)

## 🐳 Docker 배포

### 1. Dockerfile 생성

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 포트 노출
EXPOSE 8501

# 헬스체크
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 애플리케이션 실행
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Docker 이미지 빌드

```bash
docker build -t krw-momentum-radar .
```

### 3. Docker 컨테이너 실행

```bash
docker run -p 8501:8501 krw-momentum-radar
```

### 4. Docker Compose 사용

```yaml
# docker-compose.yml
version: "3.8"

services:
  momentum-radar:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## ☁️ 클라우드 플랫폼 배포

### AWS EC2

1. **EC2 인스턴스 생성**: Ubuntu 20.04+ 권장
2. **보안 그룹 설정**: 포트 8501 열기
3. **애플리케이션 배포**:

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python 설치
sudo apt install python3 python3-pip -y

# 애플리케이션 클론
git clone <your-repo-url>
cd Momentum

# 의존성 설치
pip3 install -r requirements.txt

# Streamlit 실행
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

4. **Nginx 리버스 프록시 설정** (선택사항):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Google Cloud Platform

1. **Cloud Run 사용**:

```bash
# Docker 이미지 빌드
gcloud builds submit --tag gcr.io/PROJECT-ID/krw-momentum-radar

# Cloud Run에 배포
gcloud run deploy --image gcr.io/PROJECT-ID/krw-momentum-radar --platform managed --region asia-northeast1 --allow-unauthenticated
```

2. **App Engine 사용**:

```yaml
# app.yaml
runtime: python39

env_variables:
  STREAMLIT_SERVER_HEADLESS: "true"
  STREAMLIT_SERVER_PORT: "8080"
  STREAMLIT_SERVER_ADDRESS: "0.0.0.0"

handlers:
  - url: /.*
    script: auto
```

### Azure

1. **Container Instances 사용**:

```bash
# Azure CLI로 배포
az container create \
  --resource-group myResourceGroup \
  --name momentum-radar \
  --image your-registry/krw-momentum-radar \
  --ports 8501 \
  --dns-name-label momentum-radar
```

## 🔧 환경 변수 설정

### 필수 환경 변수

```bash
# Streamlit 설정
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 데이터 소스 설정 (선택사항)
YAHOO_FINANCE_CACHE_ENABLED=true
YAHOO_FINANCE_CACHE_TTL=3600
```

### 보안 설정

```bash
# Streamlit Cloud Secrets
# .streamlit/secrets.toml (로컬 개발용)
# Streamlit Cloud 대시보드에서 설정 (프로덕션용)

[api_keys]
yahoo_finance_key = "your-api-key"  # 필요시
```

## 📊 모니터링 및 로깅

### Streamlit Cloud 모니터링

- **대시보드**: Streamlit Cloud 대시보드에서 앱 상태 확인
- **로그**: 실시간 로그 및 에러 추적
- **메트릭**: 사용량 및 성능 메트릭

### 자체 호스팅 모니터링

```python
# app.py에 추가할 모니터링 코드
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 애플리케이션 시작 시
logger.info("KRW Momentum Radar started")
```

## 🔄 CI/CD 파이프라인

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Test application
        run: |
          streamlit run app.py --server.headless true --server.port 8501 &
          sleep 10
          curl -f http://localhost:8501/_stcore/health || exit 1
```

## 🚨 트러블슈팅

### 일반적인 문제

1. **빌드 실패**:

   - `requirements.txt` 의존성 확인
   - Python 버전 호환성 확인
   - 메모리 부족 시 빌드 타임아웃 증가

2. **데이터 로드 실패**:

   - 네트워크 연결 확인
   - Yahoo Finance API 제한 확인
   - 캐시 초기화

3. **성능 문제**:
   - 캐싱 설정 최적화
   - 데이터 로드 청크 크기 조정
   - 불필요한 계산 제거

### 로그 확인

```bash
# Streamlit Cloud
# 대시보드에서 실시간 로그 확인

# 자체 호스팅
tail -f app.log
```

## 📈 성능 최적화

### 캐싱 전략

```python
# app.py에서 캐싱 최적화
@st.cache_data(ttl=60*60*6)  # 6시간 캐시
def expensive_data_operation():
    # 무거운 데이터 처리
    pass
```

### 리소스 관리

- **메모리**: 불필요한 데이터 정리
- **CPU**: 계산 최적화
- **네트워크**: 데이터 다운로드 최적화

---

**배포 완료!** 🎉 이제 KRW Momentum Radar를 전 세계에서 사용할 수 있습니다!
