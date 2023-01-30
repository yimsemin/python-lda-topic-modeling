""" 1줄에 1개의 문서가 담긴 엑셀파일에서 각 문서를 전처리하여 새로운 시트에 추가

1단계 OKT 기반 명사화
2단계 불용어 제거 (자체 사전 적용 가능 -> custom_okt 참고
3단계 1글자 단어 제거
4단계 적게 등장한 단어 제거

TODO : '적게 n개 이하의 문서에서 등장한 단어 제거' 추가
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
        'sheet_name': 0,                            # 0은 가장 앞에 있는 시트를 의미함
        'column_name': 'article',
        # 1     article         <- 제목 줄
        # 2     1줄에 1개의 문서 ...
        # 3     1줄에 1개의 문서 ...
        # 4     1줄에 1개의 문서 ...
        # ...                            ...

        # output
        'result_sheet_name': 'preprocessed',        # 시트가 이미 있으면 덮어쓰므로 주의
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
        article_series: 한 줄에 문서 하나씩

    Returns:
        (pd.Series) 한 줄에 명사만 추출된 문서 하나씩
    """
    tqdm.pandas()
    okt = Okt()

    return article_series.progress_map(lambda x: okt.nouns(x))


def remove_stop_words_from_each_article(tokenized_article_series: pd.Series) -> pd.Series:
    """ 각 열의 문서에 대해 불용어 사전 기준으로 불용어 제거

    Args:
        tokenized_article_series: 한 줄에 문서 하나씩

    Returns:
        (pd.Series) 한 줄에 불용어 제거된 문서 하나씩
    """
    tqdm.pandas()
    stop_words_set = get_stop_words()

    return tokenized_article_series.progress_map(lambda x: [word for word in x if word not in stop_words_set])


def remove_one_character_from_each_article(tokenized_article_series) -> pd.Series:
    """ 각 열의 문서에 대해 한 글자인 단어 제거

    Args:
        tokenized_article_series: 한 줄에 문서 하나씩

    Returns:
        (pd.Series) 한 줄에 한 글자 단어가 제거된 문서 하나씩
    """
    tqdm.pandas()

    return tokenized_article_series.progress_map(lambda line: [word for word in line if len(word) > 1])


def remove_low_count_word(tokenized_article_series, min_word_count: int = 20) -> pd.Series:
    """ 각 열의 문서에 대해 적게 등장한 단어 제거

    Args:
        tokenized_article_series:
        min_word_count:

    Returns:
        (pd.Series)
    """
    # TODO 아에 삭제가 되는 문서에 대한 처리방안 고민 필요 / 결측치가 되어버려서 그러함
    # TODO 데이터가 많아지면 상당히 느려짐 / 멀티프로세싱 알아보면 좋을 듯함

    word_count_series = tokenized_article_series.explode().value_counts()
    word_to_delete = set(word_count_series[word_count_series <= min_word_count].index.tolist())

    return pd.Series([[i for i in article if i not in word_to_delete] for article in tqdm(tokenized_article_series)])


def preprocessing_noun(setting: dict = None, article_series: pd.Series = None):
    """

    Args:
        setting:
        article_series:

    Returns:
        (pd.Series) 명사 추출 -> 불용어 제거 -> 한글자 제거 -> 적게 등장한 글자 제거
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
    openxlsx.save_to_xlsx(tokenized_article_series,
                          setting['xlsx_name'],
                          setting['result_column_name'],
                          setting['result_sheet_name'])


def main():
    with recorder.WithTimeRecorder('lda_explore_topic_number'):
        preprocessing_noun()


if __name__ == '__main__':
    main()
