""" 1줄에 1개의 문서가 담긴 엑셀파일에서 각 문서를 전처리하여 새로운 시트에 추가

1단계 OKT 기반 명사화
2단계 불용어 제거 (자체 사전 적용 가능 -> custom_okt 참고)
3단계 1글자 단어 제거
4단계 적게 등장한 단어 제거

TODO : '적게 n개 이하의 문서에서 등장한 단어 제거' 추가 / 전처리 단계에서 제거할지 혹은 corpus 구성한 다음 지울지 판단 필요
"""
import pandas as pd
from tqdm import tqdm

from konlpy.tag import Okt

import util.recorder as recorder
import util.openxlsx as openxlsx
from stopwords.stopwordlist import get_stop_words


def _setting():
    setting = {
        # input - 전처리를 수행할 엑셀파일
        'xlsx_name': 'input/test.xlsx',
        'sheet_name': 0,                            # 시트 이름 str 입력 / 0 입력시 가장 왼쪽에 있는 시트를 선택함
        'column_name': 'article',
        # 1     article         <- 제목 줄
        # 2     1줄에 1개의 문서 ...
        # 3     1줄에 1개의 문서 ...
        # 4     1줄에 1개의 문서 ...
        # ...                            ...

        # output
        'result_sheet_name': 'preprocessed',        # 시트가 이미 존재하면 덮어씀
        'result_column_name': 'article',
        'min_word_count': 50
    }

    article_series = openxlsx.load_series_from_xlsx(setting['xlsx_name'],
                                                    setting['column_name'],
                                                    setting['sheet_name'])

    return setting, article_series


def extract_noun_for_each_article(article_series: pd.Series) -> pd.Series:
    """ 각 열의 문서에 대해 okt(Open Korean Text) 기반으로 명사만 추출

    Args:
        article_series(pd.Series): 한 줄에 문서 하나씩

    Returns:
        (pd.Series) 한 줄에 명사만 추출된(토큰화된) 문서 하나씩
    """
    tqdm.pandas()
    okt = Okt()

    return article_series.progress_map(lambda x: okt.nouns(x))


def remove_stop_words_from_each_article(tokenized_article_series: pd.Series) -> pd.Series:
    """ 각 열의 문서에 대해 불용어 사전 기준으로 불용어 제거

    Args:
        tokenized_article_series(pd.Series): 한 줄에 토큰화된 문서 하나씩

    Returns:
        (pd.Series) 한 줄에 불용어 제거된 문서 하나씩
    """
    tqdm.pandas()
    stop_words_set = get_stop_words()       # 불용어 사전 불러오기

    return tokenized_article_series.progress_map(lambda x: [word for word in x if word not in stop_words_set])


def remove_one_character_from_each_article(tokenized_article_series) -> pd.Series:
    """ 각 열의 문서에 대해 한 글자인 단어 제거

    Args:
        tokenized_article_series(pd.Series): 한 줄에 토큰화된 문서 하나씩

    Returns:
        (pd.Series) 한 줄에 한 글자 단어가 제거된 문서 하나씩
    """
    tqdm.pandas()

    return tokenized_article_series.progress_map(lambda line: [word for word in line if len(word) > 1])


def remove_low_count_word(tokenized_article_series, min_word_count: int = 20) -> pd.Series:
    """ 각 열의 문서에 대해 적게 등장한 단어 제거

    Args:
        tokenized_article_series(pd.Series): 한 줄에 토큰화된 문서 하나씩
        min_word_count: 적게 등장한 단어를 제거할 때 그 기준, 0 입력시 진행하지 않음

    Returns:
        (pd.Series) 한 줄에 적게 등장한 단어가 제거된 문서 하나씩
    """
    # TODO 모든 단어가 삭제되어 문서가 비어버리는 경우에 대한 처리방안 고민 필요
    # TODO 데이터가 많아지면 상당히 느려짐 / 멀티프로세싱 알아보면 좋을 듯함

    # 최소 카운트가 0일 경우 함수 생략
    if min_word_count == 0:
        print('-- 적게 등장한 단어 제거의 기준이 0으로 입력되어, 제거가 진행되지 않았습니다.')
        return tokenized_article_series

    else:
        word_count_series = tokenized_article_series.explode().value_counts()
        word_to_delete = set(word_count_series[word_count_series <= min_word_count].index.tolist())

        return pd.Series([[i for i in article if i not in word_to_delete] for article in tqdm(tokenized_article_series)])


def preprocessing_noun(setting: dict = None, article_series: pd.Series = None):
    """ pd.Series 데이터를 전처리하여 xlsx 파일에 저장

    수행하는 전처리: 명사 추출 -> 불용어 제거 -> 한글자 제거 -> 적게 등장한 글자 제거

    Args:
        setting: 설정값 불러오기
            setting['xlsx_name']: str = 전처리 결과를 저장할 엑셀파일, 예: 'where/filename.xlsx'
            setting['result_sheet_name']: str = 전처리 결과를 저장할 시트 이름, 예: 'proprecessed'
            setting['result_column_name']: str = 전처리 결과를 저장할 때 머리줄에 넣을 키워드, 예: 'article'
            setting['min_word_count']: int = 적게 등장한 단어를 제거할 때 그 기준, 0 입력시 진행하지 않음
        article_series: 한 줄에 문서 하나씩
    """
    if setting or article_series is None:
        setting, article_series = _setting()

    # preprocess - Noun
    print(f'1단계: 명사를 추출합니다.')
    tokenized_article_series = extract_noun_for_each_article(article_series)
    print(f'2단계: 불용어를 제거합니다.')
    tokenized_article_series = remove_stop_words_from_each_article(tokenized_article_series)
    print(f'3단계: 한글자 단어를 제거합니다.')
    tokenized_article_series = remove_one_character_from_each_article(tokenized_article_series)
    print(f'4단계: 적게 등장한 단얼르 제거합니다.')
    tokenized_article_series = remove_low_count_word(tokenized_article_series, setting['min_word_count'])

    # save result
    with pd.ExcelWriter(setting['xlsx_name'], mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        tokenized_article_series.to_excel(writer, sheet_name=setting['result_sheet_name'])


def main():
    with recorder.WithTimeRecorder('전처리'):
        preprocessing_noun()


if __name__ == '__main__':
    main()
