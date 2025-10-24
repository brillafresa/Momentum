# 1. 프로젝트 폴더로 이동 (스크립트가 다른 위치에서 실행될 경우를 대비)
# 이 스크립트가 실행되는 폴더를 기준으로 작업합니다.
Set-Location -Path (Split-Path -Parent $PSCommandPath)

# 2. 가상 환경 활성화 및 Streamlit 실행
# cmd /c를 사용하여 activate.bat을 실행하고, 이어서 Streamlit 명령을 실행합니다.
# 주의: 가상 환경의 환경 변수 설정이 다음 명령어에 영향을 미치도록 명령을 합쳐야 합니다.
cmd /c ".\venv\Scripts\activate.bat & streamlit run app.py & deactivate"

