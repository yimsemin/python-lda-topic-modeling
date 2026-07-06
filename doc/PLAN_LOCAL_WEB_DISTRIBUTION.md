# Local Web Distribution

이 문서는 아직 구현하지 않은 Windows용 로컬 웹 배포 계획이다.

## 목적

로컬 웹 UI를 기존 CLI 위에 얇게 얹어, 코드를 직접 수정하기 어려운 사용자가 브라우저에서 분석을 실행할 수 있게 한다.

웹 UI는 기존 CLI를 대체하지 않는 비개발자용 보조 진입점이다. 기존 파일명, 설정 함수, 실행 방식, 결과 파일명, 저장 위치, CSV 컬럼명은 가능한 한 유지한다.

## 결정

- 지원 플랫폼은 우선 Windows로 제한한다.
- 사용자는 GitHub 저장소를 ZIP으로 다운로드하고 압축을 푼다. Git 사용은 요구하지 않는다.
- `initial_setting.bat`은 프로젝트 폴더 안에 실행환경을 만든다.
- `start.bat`은 로컬 웹 UI를 열고, 분석은 로컬 Python 서버가 수행한다.
- 입력 파일과 결과물은 기존 `input/`, `output/` 폴더를 사용한다.
- 입력, 중간 산출물, 결과물은 외부 서버로 업로드하지 않는다.
- 삭제는 프로젝트 폴더 전체 삭제로 끝나는 것을 목표로 한다.

이 방식은 포터블 앱 배포보다 최초 ZIP 크기가 작고, 순수 브라우저 앱보다 기존 Python 분석 코드를 더 많이 재사용할 수 있다. 대신 첫 설정에는 인터넷 연결이 필요할 수 있다.

## 흐름

```text
ZIP 다운로드
-> 압축 해제
-> initial_setting.bat 실행
-> start.bat 실행
-> 브라우저에서 로컬 웹 UI 사용
-> 필요하면 input/output 폴더 백업
-> 프로젝트 폴더 삭제
```

초기 설정 스크립트는 관리자 권한을 요구하지 않는다. 시스템 Python, 전역 PATH, 레지스트리를 바꾸지 않는다.

## 목표 구조

```text
project-root/
  initial_setting.bat
  start.bat
  .python-version
  requirements.txt

  app/
    server.py
    web/
      index.html
      assets/

  input/
  output/

  .runtime/
    uv/
    python/
    uv-cache/
    logs/
    matplotlib/

  .venv/
```

`input/`과 `output/`은 기존 CLI와 웹 UI가 함께 사용하는 사용자 데이터 영역이다. `.runtime`과 `.venv`는 내부 실행환경이므로 사용자가 직접 만질 필요가 없다.

## 실행환경 원칙

- `uv`로 Python과 패키지 설치를 자동화한다.
- `uv` 자체도 전역 설치하지 말고 `.runtime/uv`에 둔다.
- 의존성 기준은 `requirements.txt`로 유지하고 `uv.lock`은 사용하지 않는다.
- 가능한 저장 위치를 프로젝트 내부로 고정한다: `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_PROJECT_ENVIRONMENT`.
- `start`는 가능하면 `uv run`이 아니라 `.venv`의 Python을 직접 실행한다.
- 로컬 서버는 `127.0.0.1`에만 바인딩한다.
- 웹 UI의 CSS, JavaScript, 이미지, 폰트는 가능한 한 저장소 내부 자산을 사용한다.
- 설치 스크립트는 여러 번 실행해도 안전해야 한다.

## 구현 전 확인

- 현재 CLI 파이프라인이 Windows에서 정상 동작하는지 확인한다.
- 기존 공개 진입점, 파일명, 저장 위치, CSV 컬럼명은 가능한 한 유지한다.
- 로컬 웹 UI는 기존 스크립트 위에 얇게 얹고, 기존 CLI 사용 방식은 깨지 않게 한다.
- 설치와 실행 후 프로젝트 폴더 밖에 생성되는 파일이 없는지 확인한다.
- 업데이트/삭제 안내에서는 `input/`과 `output/` 백업 필요성을 명확히 한다.
