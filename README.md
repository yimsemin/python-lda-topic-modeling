# python-lda-topic-modeling

한국어 토픽모델링(Topic Modeling)을 위한 python 코드입니다. 모델링을 위해 [Gensim](https://github.com/RaRe-Technologies/gensim) 을, 한국어 텍스트 처리를 위해 [knolpy](https://github.com/konlpy/konlpy) 를 사용합니다.





## 1. 주요기능

1. 텍스트 전처리 `preprocessing.py`

   - knolpy의 Okt(Open-Korean-text) 기반 명사화 (커스텀 사전 추가 가능)
   - 사전 기반 불용어 제거 (불용어 사전은 `stopwords/stopwordlist.txt`에 1줄에 1단어씩 작성)
   - 1글자 단어 제거
   - 적게 등장한 단어 제거

2. 빈도수 분석 `frequency_analysis.py`

   - 토큰(단어)와 빈도수를 내림차순으로 제시 -> csv로 저장
   - 워드클라우드 -> png로 저장

3. 토픽 갯수 k 정하기 `lda_explore_topic_number.py`

   - 토픽 갯수 범위를 지정, 각 갯수별 lda modeling을 수행
   - 각 모델의 혼란도(perplexity)와 응집도(coherence)를 계산
   - 결과값을 csv로, 이를 시각화한 그래프를 png로 저장

4. LDA `lda.py`

   - 전처리된 분석 대상 문서의 말뭉치(corpus)와 딕셔너리(dictionary)의 생성, 저장
   - 지정된 토픽 갯수의 LDA 모델 생성
   - 각 문서의 토픽 분포제시 -> csv로 저장
   - 생성된 LDA 모델의 시각화 -> html로 저장

5. 시간의 흐름에 따른 토픽 논의 추세를 파악하기 위한 회귀분석 기반 시계열 분석 `lda_hot_and_cold.py`

   - 각 문서의 토픽 별 θ 값 계산
   - y(θ) = ax(time) + b 의 선형 회귀분석
   - Hot & Cold 토픽 제시

6. 기타 기능들...
   - okt 사전에 커스텀 사전(명사, 오탈자) 추가 `custom_okt/okt_add_custom_dict.py`





## 2. 실행환경

### 주요 패키지 버전

- 코드를 테스트했던 버전임
- `python == 3.10.9`
- `gensim == 4.3.0`
- `knolpy == 0.6.0`
- `pandas == 1.5.2`
- `pyldavis == 3.3.1`
- `statsmodels == 0.13.5`
- `wordcloud == 1.9.3`



### Java

- knolpy 사용을 위해서는 java가 필요하나, 최신 버전의 java에서 작동이 안되는 경우가 있습니다.
- 맥의 경우 : zulu JDK 17
  - https://www.azul.com/downloads/?version=java-17-lts&os=macos&architecture=arm-64-bit&package=jdk
  - M1 pro에서 동작 확인함
- 윈도우의 경우 : JDK 17 - java version "17.0.4.1" 2022-08-18 LTS
  - https://www.oracle.com/java/technologies/downloads/#java17
  - Win 10에서 동작 확인함





## 3. 사용방법

1. 코드를 다운로드하고, 코드를 실행시킬 수 있는 환경을 구성합니다(상단 실행환경 참조)
2. raw 데이터의 입력 → 엑셀파일(.xlsx)
3. 상단의 `_setting()` 에서 세부 설정을 조정할 수 있음
4. 결과물이 생성될 폴더는 미리 만들어둬야 함
