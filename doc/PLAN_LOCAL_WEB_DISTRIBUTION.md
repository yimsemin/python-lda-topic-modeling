# Local Web Distribution

이 문서는 아직 구현하지 않은 향후 배포 계획이다.

## 결정

- 사용자는 GitHub 저장소를 ZIP으로 다운로드하고 압축을 푼다. Git 사용은 요구하지 않는다.
- 첫 실행은 `initial_setting` 스크립트가 프로젝트 폴더 안에 실행환경을 만든다.
- 이후 실행은 `start` 스크립트가 로컬 웹 UI를 열고, 분석은 로컬 Python 서버가 수행한다.
- 입력, 중간 산출물, 결과물은 외부 서버로 업로드하지 않는다.
- 삭제는 프로젝트 폴더 전체 삭제로 끝나는 것을 목표로 한다.

이 방식은 포터블 앱 배포보다 최초 ZIP 크기가 작고, 순수 브라우저 앱보다 기존 Python 분석 코드를 더 많이 재사용할 수 있다. 대신 첫 설정에는 인터넷 연결이 필요할 수 있다.

## 사용자 흐름

```text
ZIP 다운로드
-> 압축 해제
-> initial_setting 실행
-> start 실행
-> 브라우저에서 로컬 웹 UI 사용
-> 필요하면 workspace 폴더 백업
-> 프로젝트 폴더 삭제
```

플랫폼별 진입점:

- Windows: `initial_setting.bat`, `start.bat`, 필요하면 `stop.bat`
- macOS: `initial_setting.command`, `start.command`, 필요하면 `stop.command`

초기 설정 스크립트는 관리자 권한이나 `sudo`를 요구하지 않는다. 시스템 Python, 전역 PATH, 레지스트리, 사용자 shell profile을 바꾸지 않는다.

## 목표 구조

```text
project-root/
  initial_setting.bat
  start.bat
  initial_setting.command
  start.command
  pyproject.toml
  uv.lock
  .python-version

  app/
    server.py
    web/
      index.html
      assets/

  workspace/
    input/
    intermediate/
    output/

  .runtime/
    uv/
    python/
    uv-cache/
    logs/
    matplotlib/

  .venv/
```

`workspace`는 사용자 데이터 영역이다. `.runtime`과 `.venv`는 내부 실행환경이므로 사용자가 직접 만질 필요가 없다.

## 실행환경 원칙

- `uv`로 Python과 패키지 설치를 자동화한다.
- `uv` 자체도 전역 설치하지 말고 `.runtime/uv`에 둔다.
- 다음 저장 위치를 프로젝트 내부로 고정한다: `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_PROJECT_ENVIRONMENT`, `UV_TOOL_DIR`, `UV_TOOL_BIN_DIR`.
- `start`는 가능하면 `uv run`이 아니라 `.venv`의 Python을 직접 실행한다.
- 로컬 서버는 `127.0.0.1`에만 바인딩한다.
- 웹 UI의 CSS, JavaScript, 이미지, 폰트는 외부 CDN이 아니라 저장소 내부 자산을 사용한다.
- 설치 스크립트는 여러 번 실행해도 안전해야 한다.

## macOS 주의점

- `.command` 파일은 Finder 더블클릭 실행을 위해 실행 권한이 필요하다.
- GitHub ZIP 다운로드 후 실행 권한이 유지되는지 실제 macOS에서 확인한다.
- 실행 권한이 유지되지 않으면 비개발자용 대체 실행 방법을 다시 설계한다.
- Apple Silicon과 Intel Mac을 구분해 올바른 uv, Python, wheel이 설치되는지 확인한다.
- Gatekeeper 또는 보안 경고가 뜰 수 있다. 실제 구현 후 README에 필요한 최소 안내만 추가한다.

## 구현 전 확인

- 현재 CLI 파이프라인이 Windows와 macOS에서 정상 동작하는지 확인한다.
- 기존 공개 진입점, 파일명, 저장 위치, CSV 컬럼명은 가능한 한 유지한다.
- 로컬 웹 UI는 기존 스크립트 위에 얇게 얹고, 기존 CLI 사용 방식은 깨지 않게 한다.
- 설치와 실행 후 프로젝트 폴더 밖에 생성되는 파일이 없는지 확인한다.
- 업데이트/삭제 안내에서는 `workspace` 백업 필요성을 명확히 한다.
