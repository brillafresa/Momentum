# 기여 가이드

KRW Momentum Radar 프로젝트에 기여해주셔서 감사합니다! 이 문서는 프로젝트에 기여하는 방법을 안내합니다.

## 🚀 시작하기

### 개발 환경 설정

1. 저장소를 포크하고 클론합니다:

```bash
git clone https://github.com/your-username/Momentum.git
cd Momentum
```

2. 가상환경을 생성하고 활성화합니다 (Python 3.11 권장):

```bash
py -3.11 -m venv venv || python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성을 설치합니다:

```bash
pip install -r requirements.txt
```

4. 애플리케이션을 실행하여 정상 작동을 확인합니다:

```bash
streamlit run app.py
```

## 📝 기여 방법

### 버그 리포트

버그를 발견하셨다면 다음 정보를 포함하여 이슈를 생성해주세요:

- **버그 설명**: 무엇이 잘못되었는지 명확하게 설명
- **재현 단계**: 버그를 재현하는 단계별 방법
- **예상 결과**: 무엇이 일어나야 하는지
- **실제 결과**: 실제로 무엇이 일어났는지
- **환경 정보**: OS, Python 버전, 브라우저 등

### 기능 제안

새로운 기능을 제안하고 싶으시다면:

- **기능 설명**: 어떤 기능을 원하는지 자세히 설명
- **사용 사례**: 이 기능이 어떻게 유용할지
- **구현 아이디어**: 가능하다면 구현 방법에 대한 아이디어

### 코드 기여

1. **브랜치 생성**: 새로운 기능이나 버그 수정을 위한 브랜치를 생성합니다:

```bash
git checkout -b feature/your-feature-name
# 또는
git checkout -b fix/your-bug-fix
```

2. **코드 작성**: 변경사항을 구현합니다.

3. **테스트**: 로컬에서 애플리케이션이 정상 작동하는지 확인합니다.

4. **커밋**: 의미있는 커밋 메시지와 함께 변경사항을 커밋합니다:

```bash
git add .
git commit -m "Add: 새로운 기능 추가"
```

5. **푸시**: 브랜치를 원격 저장소에 푸시합니다:

```bash
git push origin feature/your-feature-name
```

6. **Pull Request 생성**: GitHub에서 Pull Request를 생성합니다.

## 📋 코딩 스타일

### Python 스타일 가이드

- **PEP 8** 준수
- **함수/변수명**: snake_case 사용
- **클래스명**: PascalCase 사용
- **상수**: UPPER_CASE 사용
- **문서화**: 함수와 클래스에 docstring 추가

### 예시:

```python
def calculate_momentum_score(prices_df, window=21):
    """
    모멘텀 스코어를 계산합니다.

    Args:
        prices_df (pd.DataFrame): 가격 데이터
        window (int): 계산 윈도우 (기본값: 21)

    Returns:
        pd.Series: 모멘텀 스코어
    """
    # 구현 코드
    pass
```

### Streamlit 스타일

- **세션 상태**: `st.session_state` 적절히 활용
- **캐싱**: `@st.cache_data` 적절히 사용
- **UI 구성**: 일관된 레이아웃과 스타일 유지
- **에러 처리**: 사용자 친화적인 에러 메시지

## 🧪 테스트

### 로컬 테스트

기여하기 전에 다음을 확인해주세요:

1. **기능 테스트**: 모든 기능이 정상 작동하는지
2. **UI 테스트**: 다양한 화면 크기에서 UI가 올바르게 표시되는지
3. **데이터 테스트**: 다양한 시장 상황에서 데이터가 올바르게 로드되는지
4. **성능 테스트**: 애플리케이션이 적절한 속도로 실행되는지

### 테스트 체크리스트

- [ ] 새로운 기능이 기존 기능을 깨뜨리지 않는가?
- [ ] 에러 처리가 적절한가?
- [ ] 사용자 인터페이스가 직관적인가?
- [ ] 코드가 읽기 쉽고 유지보수 가능한가?
- [ ] 문서화가 충분한가?

## 📚 문서화

### 코드 문서화

- 모든 함수와 클래스에 docstring 추가
- 복잡한 로직에는 주석 추가
- 타입 힌트 사용 권장

### README 업데이트

새로운 기능을 추가할 때는 README.md를 업데이트해주세요:

- 새로운 기능 설명
- 사용법 예시
- 설정 옵션 변경사항

## 🔄 Pull Request 프로세스

### PR 생성 시 포함할 내용

1. **제목**: 변경사항을 명확하게 설명
2. **설명**:
   - 변경사항 요약
   - 변경 이유
   - 테스트 방법
   - 스크린샷 (UI 변경 시)

### PR 리뷰 과정

1. **자동 검사**: CI/CD 파이프라인 통과
2. **코드 리뷰**: 프로젝트 메인테이너의 리뷰
3. **테스트**: 기능 테스트 및 통합 테스트
4. **병합**: 승인 후 main 브랜치에 병합

## 🏷️ 릴리스

### 버전 관리

- **Major**: 호환성을 깨는 변경
- **Minor**: 새로운 기능 추가
- **Patch**: 버그 수정

### 릴리스 노트

각 릴리스마다 다음을 포함합니다:

- 새로운 기능
- 버그 수정
- 성능 개선
- Breaking changes

## 💬 커뮤니티

### 질문과 토론

- **GitHub Issues**: 버그 리포트 및 기능 제안
- **GitHub Discussions**: 일반적인 질문과 토론
- **Pull Request**: 코드 리뷰 및 토론

### 행동 강령

모든 기여자는 다음을 준수해야 합니다:

- 존중과 예의
- 건설적인 피드백
- 다양성과 포용성
- 협력적인 태도

## 📞 연락처

프로젝트에 대한 질문이나 제안이 있으시면:

- GitHub Issues를 통해 문의
- Pull Request를 통한 기여
- GitHub Discussions에서 토론

---

**감사합니다!** KRW Momentum Radar를 더 나은 도구로 만들어주셔서 감사합니다. 🚀
