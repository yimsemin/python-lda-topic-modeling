# python-lda-topic-modeling

한국어 토픽모델링(Topic Modeling)을 위한 python 코드입니다. 모델링에는 [Gensim](https://github.com/RaRe-Technologies/gensim)을, 한국어 텍스트 처리에는 [kiwipiepy](https://github.com/bab2min/kiwipiepy)를 사용합니다.
또한 워드클라우드 한글 렌더링을 위해 `font/NanumGothic.ttf`를 포함하며, 라이선스 전문은 `font/OFL.txt`를 참조합니다.

`run_analysis.py`의 `_setting()`만 수정하여 실행할 수 있습니다.

## 1. 실행환경

### Windows 초기 설정

GitHub ZIP을 다운로드해 압축을 푼 뒤 `initial_setting.bat`을 실행하여 실행환경을 만들 수 있습니다.

이 스크립트는 Windows 기본 도구인 `curl.exe`와 `tar.exe`로 `uv`를 `.runtime/uv`에 내려받고, `.python-version`의 Python 버전으로 프로젝트 내부 Python과 `.venv`를 만든 뒤, `requirements.txt`의 패키지를 설치합니다. 관리자 권한, 전역 PATH 변경, 시스템 Python 설치를 요구하지 않습니다.

재설치가 필요할 경우 `.venv` 및 `.runtime` 폴더를 삭제한 뒤 다시 실행합니다.

### macOS 수동 설정

GitHub ZIP을 다운로드해 압축을 푼 뒤 해당 폴더로 이동합니다.

```bash
cd ~/Downloads/python-lda-topic-modeling
```

가상환경을 설정합니다. 필요한 Python 버전은 `.python-version`에서 확인합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p input output
```

### 가상환경 내 코드 실행

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python run_analysis.py
```

Windows 명령 프롬프트(cmd):

```bat
.venv\Scripts\activate.bat
python run_analysis.py
```

macOS 터미널(terminal):

```bash
source .venv/bin/activate
python run_analysis.py
```

## 2. 입력 파일 준비

- 엑셀 파일(`.xlsx`)은 한 행에 문서 하나가 들어가는 형태로 준비합니다.
- 원문 시트에는 분석 대상 문서 열이 있어야 합니다. 기본 열 이름은 `article`입니다.
- 시계열 분석을 할 경우 원문 시트에 날짜 열도 준비합니다. 기본 열 이름은 `date`입니다.
- 날짜는 엑셀 날짜 서식, serial date, `YYYYMMDD`, `YYMMDD`, `YYYY-MM-DD`, `YYYY/MM/DD` 형식을 지원합니다.
- 사용자 정의 불용어는 `input/stopwordlist.example.txt`를 참고해 `input/stopwordlist.txt`에 작성합니다. 한 줄에 `단어/품사` 형식으로 쓰며, 품사를 생략하면 `NNG`로 처리합니다.

## 3. run_analysis.py 실행

`run_analysis.py`의 `_setting()`에서 입력, 출력, 전처리, 토픽 수 탐색, 최종 모델, 캐시 설정을 한 번에 관리합니다.

터미널에서는 보통 `run_analysis.py`를 실행합니다. `_setting()`에는 실행에 필요한 설정값이 모여 있고, `tasks`에 적힌 번호에 따라 `preprocessing.py`, `frequency_analysis.py`, `lda_explore_topic_number.py`, `lda.py`, `lda_hot_and_cold.py`의 기능이 차례대로 실행됩니다.

`tasks`에는 실행할 작업 번호를 순서대로 넣습니다.

| 번호 | 실행 작업 |
| --- | --- |
| `1` | 전처리 |
| `2` | 빈도분석 |
| `3` | 토픽 갯수 탐색 |
| `4` | 선택한 토픽 수로 LDA 모델 반복 생성 |
| `5` | 선택한 LDA 모델로 Hot/Cold 시계열 분석 |

예를 들어 전처리만 실행하려면 `tasks`를 `[1]`로 두고, 전처리 후 토픽 수 탐색까지 이어서 실행하려면 `[1, 2, 3]`처럼 둡니다.

자주 수정하는 설정값은 아래와 같습니다.

| 설정값 | 의미 |
| --- | --- |
| `tasks` | 실행할 작업 번호 목록입니다. 처음에는 `[1]`로 전처리만 확인하는 것을 권장합니다. |
| `xlsx_name` | 분석할 엑셀 파일 경로입니다. 기본값은 `input/data.xlsx`입니다. |
| `raw_sheet_name` | 원문이 들어 있는 시트입니다. `0`이면 가장 왼쪽 시트를 사용합니다. |
| `text_column_name` | 분석할 본문 열 이름입니다. 기본값은 `article`입니다. |
| `stopwordlist_location` | 사용자 불용어 파일 경로입니다. 기본값은 `input/stopwordlist.txt`입니다. |
| `preprocessing_min_word_count` | 너무 적게 등장한 단어를 제거하는 기준입니다. |
| `topic_number_start`, `topic_number_end` | 토픽 수 후보를 탐색할 범위입니다. |
| `num_topics` | 최종 모델을 만들 때 사용할 토픽 수입니다. |
| `reuse_saved_corpus`, `reuse_saved_model` | 이전 실행 결과를 재사용할지 정합니다. 입력이나 전처리 기준을 바꿨다면 `False`로 두거나 기존 결과 폴더를 비웁니다. |

가상환경을 활성화했다면 아래처럼 실행합니다.

```bash
python run_analysis.py
```

## 4. 권장 사용자 워크플로우

1. 원문 엑셀 파일을 준비합니다.
   - `run_analysis.py`의 `_setting()`에서 `xlsx_name`, `raw_sheet_name`, `text_column_name`, `stopwordlist_location`, `preprocessing_min_word_count`를 확인합니다.

2. 전처리를 실행합니다.
   - `tasks`: `[1]`
   - 결과는 같은 엑셀 파일의 `preprocessed_sheet_name` 시트에 저장됩니다. 기본값은 `preprocessed`입니다.
   - 빈 문서는 삭제하지 않고 저장합니다. 대신 로그와 `empty_document_csv_name` 파일에서 빈 문서의 `pandas_index`, 엑셀 행 번호, 원문 미리보기를 확인할 수 있습니다.

3. 빈 문서를 검토하고 입력을 조정합니다.
   - 빈 문서가 생기면 원문 엑셀에서 해당 행을 삭제하거나 원문을 보완합니다.
   - 너무 많은 단어가 제거된 경우 `min_word_count`를 낮춥니다. 이 값은 `n회 이하 등장한 단어를 제거`하는 기준입니다.
   - 불필요한 단어가 남는 경우 사용자 정의 불용어를 추가합니다.
   - 빈 문서가 없어질 때까지 전처리를 반복합니다.

4. 이전 산출물이 새 입력과 섞이지 않도록 정리합니다.
   - 입력 엑셀이나 전처리 기준을 바꾼 뒤 LDA를 다시 돌릴 때는 기존 `result_dir`와 `result_model_dir` 결과를 삭제하거나 새 결과 폴더를 사용합니다.
   - 삭제 대신 설정으로 새로 만들려면 `reuse_saved_corpus=False`로 둡니다.
   - 입력이 바뀐 재분석에서는 `reuse_saved_model=False`도 함께 둡니다.

5. 필요하면 빈도분석으로 전처리 결과를 점검합니다.
   - `tasks`: `[2]`
   - 결과: `frequency_analysis.csv`, `word_cloud.png`
   - 너무 일반적인 단어가 상위에 남으면 불용어를 보완하고 전처리부터 다시 실행합니다.

6. 적절한 토픽 수 후보를 탐색합니다.
   - `tasks`: `[3]`
   - `topic_number_start`, `topic_number_end`, `topic_number_interval`로 탐색 범위를 정합니다.
   - 불연속 후보만 보고 싶으면 `topic_number_list`에 직접 추가할 수 있습니다. 예: `[5, 7, 10, 12]`
   - 결과: `lda__explore_topic_number.csv`, `lda__perplexity_value.png`, `lda__coherence_value.png`
   - `save_explore_html=True`이면 각 k별 HTML을 저장합니다.
   - `save_explore_topic_detail=True`이면 각 k별 토픽 키워드 CSV와 문서별 토픽 분포 CSV도 저장합니다.

7. 사람이 최종 토픽 수를 선택합니다.
   - perplexity는 낮을수록, coherence는 높을수록 참고 가치가 있습니다.
   - 지표만으로 결정하지 말고 k별 HTML, 토픽 키워드 CSV, 문서별 토픽 분포 CSV를 함께 봅니다.
   - 주요 키워드가 여러 토픽에 과도하게 중복되지 않는지, 특정 주제가 지나치게 쪼개지거나 뭉개지지 않는지, 토픽별 해석이 가능한지를 기준으로 선택합니다.

8. 선택한 토픽 수로 여러 랜덤 바리에이션을 확인합니다.
   - `tasks`: `[4]`
   - `num_topics`에 선택한 토픽 수를 입력합니다.
   - `task_repeat=5`이면 `random_state`를 1씩 증가시키며 같은 토픽 수의 모델 5개를 만듭니다.
   - 결과 파일은 `lda_k_{토픽수}_rd_{랜덤값}` 이름으로 저장됩니다.
   - HTML, 토픽 키워드 CSV, 문서별 토픽 분포 CSV를 비교해 토픽 구조가 크게 흔들리지 않는지 확인합니다.

9. 최종 분석 모델을 선택합니다.
   - 선택한 모델 이름을 기준으로 HTML, 토픽 키워드 CSV, 문서별 토픽 분포 CSV를 해석에 사용합니다.
   - 시계열 분석을 할 경우 `lda_model`에 선택한 모델 경로를 입력하고 `tasks`를 `[5]`로 둡니다.

## 5. 주요 파일과 산출물

| 파일 | 역할 | 주요 산출물 |
| --- | --- | --- |
| `run_analysis.py` | 통합 실행 파일 | 선택한 task 번호에 따라 아래 산출물 생성 |
| `preprocessing.py` | 명사 추출, 불용어 제거, 저빈도 단어 제거 | 엑셀 `preprocessed` 시트, `preprocessing_empty_documents.csv` |
| `frequency_analysis.py` | 전처리 결과의 단어 빈도 확인 | `frequency_analysis.csv`, `word_cloud.png` |
| `lda_explore_topic_number.py` | 토픽 수 후보별 모델 생성과 지표 계산 | `lda__explore_topic_number.csv`, 지표 그래프, k별 HTML/CSV |
| `lda.py` | 선택한 토픽 수로 최종 후보 모델 반복 생성 | 모델 파일, HTML, 토픽 키워드 CSV, 문서별 토픽 분포 CSV |
| `lda_hot_and_cold.py` | 날짜 흐름에 따른 토픽 증감 분석 | `time_and_theta.csv`, `hot_and_cold.csv`, `hot_and_cold.png` |

## 6. 캐시와 재실행 주의사항

`dictionary`, `corpus`, `model/` 파일은 실행 시간을 줄이기 위해 재사용됩니다. 같은 입력과 같은 설정을 반복 실행할 때는 유용하지만, 입력 엑셀이나 전처리 기준이 바뀐 뒤에는 새로 생성해야 합니다.

새 입력으로 다시 분석할 경우 다음 선택지 중 하나를 선택해서 진행해주세요.

1. (추천) 기존 결과 폴더를 비우고 다시 실행합니다.
2. 새 `result_dir`, `result_model_dir`를 지정합니다.
3. `reuse_saved_corpus=False`, `reuse_saved_model=False`로 설정해 새로 생성합니다.

## 7. 테스트 데이터

샘플로 제공되는 테스트 데이터(`test/input/data.xlsx`)는 AI로 생성한 가상의 B2B SaaS 기업 재직자 1:1 반구조화 면담 응답 140건입니다. 면담 내용은 조직개편, 성과평가, 핵심 인력 이탈 시나리오를 중심으로 구성했으며, 원본 생성 지침과 토픽 사전은 [doc/TEST_DATA_GENERATION_SCENARIO.md](doc/TEST_DATA_GENERATION_SCENARIO.md), 데이터 기술통계는 [doc/TEST_DATA_DESCRIPTIVE_STATISTICS.md](doc/TEST_DATA_DESCRIPTIVE_STATISTICS.md)를 참조합니다.

데이터는 생성은 ChatGPT를 통해 2026년 6월 30일에 수행했으며, 한 번에 일괄 생성하지 않고 단계적으로 진행했습니다. 첫째, 시나리오와 토픽 사전을 기준으로 면담일, 부서, 직급, 근속연수, 주 토픽, 보조 토픽, 본문을 가진 응답 초안을 만들었습니다. 둘째, 초안별로 월별 사건과의 정합성, 부서·직급별 관점, 토픽 중심성, 문체 자연성, 응답 간 반복성을 검토했습니다. 셋째, 검토 결과를 바탕으로 긍정·중립·부정 경험, 간접 관찰, 모호한 감정, 덜 정돈된 발화, 회의·메신저·고객 통화·평가 면담 같은 구체적 업무 장면을 보강했습니다. 넷째, 누적된 데이터셋 전체에서 중복·근접 중복, 날짜와 사건 월의 충돌, 메타데이터와 본문의 불일치, 날짜 형식과 토픽 구분자 혼재를 점검했습니다.
