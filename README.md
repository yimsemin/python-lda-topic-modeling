# python-lda-topic-modeling

한국어 토픽모델링(Topic Modeling)을 위한 python 코드입니다. 모델링을 위해 [Gensim](https://github.com/RaRe-Technologies/gensim) 을, 한국어 텍스트 처리를 위해 [knolpy](https://github.com/konlpy/konlpy) 를 사용합니다.





## 1. 주요기능

1. 텍스트 전처리 `preprocessing.py`

   - 문서별 토큰화
   - 불용어 제거
   - 한글자 제거
   - 적게 등장한 단어 제거

2. 빈도수 분석 `frequency_analysis.py`

   - 토큰별 등장횟수 제시
   - n개 미만 등장 토큰 제거
   - (작업 예정) 워드클라우드

3. LDA `lda.py`

   - LDA 모델 생성
   - LDA 모델을 그래프로 시각화

4. 토픽 갯수 k 정하기 `lda_get_topic_number.py`

   - 토픽 갯수 별 perplexity 계산 및 시각화
   - 토픽 갯수 별 coherence 계산 및 시각화

5. 회귀분석을 기반으로 한 시계열 LDA `lda_seq.py`

   - LDA seq 모델 생성(Dynamic Topic Modeling)
   - 각 문서의 토픽 별 θ 값 계산
   - (보완 중) time(y) = aθ(x) + b 의 회귀분석을 통한 Hot & Cold 토픽 제시

6. 기타 부가기능들...
   - 커스텀 불용어 사전 생성 `stopwords/stowrdlist.py`

   - 임시방편을 위한 정규표현식으로 수정 `preprocessing_regex.py`

   - okt 사전에 명사, 오탈자 수정 추가 `custom_okt/okt_add_custom_dict.py`





## 2. 실행환경

### 주요 패키지 버전

- `python == 3.10.9`
- `gensim == 4.3.0`
- `knolpy == 0.6.0`
- `pandas == 1.5.2`
- `pyldavis == 3.3.1`
- `statsmodels == 0.13.5`



### Java

- knolpy 사용을 위해서는 java가 필요하나, 최신 버전의 java에서 작동이 안되는 경우가 있습니다.
- 맥의 경우 : zulu JDK 17
  - https://www.azul.com/downloads/?version=java-17-lts&os=macos&architecture=arm-64-bit&package=jdk
  - M1 pro에서 동작 확인함
- 윈도우의 경우 : JDK 17 - java version "17.0.4.1" 2022-08-18 LTS
  - https://www.oracle.com/java/technologies/downloads/#java17
  - Win 10에서 동작 확인함





## 3. 사용방법

### 토픽모델링

1. 코드를 다운로드하고, 코드를 실행시킬 수 있는 환경을 구성합니다(상단 실행환경 참조)
2. raw 데이터의 입력 → 엑셀파일(.xlsx)
3. 상단의 `_setting()` 에서 세부 설정을 조정할 수 있음
4. 결과물이 생성될 폴더는 미리 만들어둬야 함
5. hot_and_cold는 보완중으로, time_and_theta.csv 가 만들어지면 이를 spss 또는 jamovi로 직접 회귀분석 돌리는 것을 추천함


