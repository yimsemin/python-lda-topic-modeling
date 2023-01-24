import pandas as pd
from tqdm import tqdm

from konlpy.tag import Okt

import util.openxlsx as openxlsx
from stopwords.stopwordlist import get_stop_words as _get_stop_words


def _setting():
    # input
    input_data = {
        'xlsx_name': 'input/test.xlsx',
        'sheet_name': 0,
        'column_name': 'article',
    }

    # output
    output_data = {
        'result_xlsx_name': 'input/test.xlsx',
        'result_sheet_name': 'preprocessed',
        'result_column_name': 'article',
        'min_word_count': 50
    }

    return input_data, output_data


def extract_noun_for_each_article(article_series):
    """ series의 각 열을 Okt() 기반으로 명사만 추출

    Args:
        article_series: 한 줄에 문서 하나씩

    Returns:
        series: 한 줄에 명사만 추출된 문서 하나씩

    """

    okt = Okt()

    try:
        result = article_series.progress_map(lambda x: okt.nouns(x))
    except AttributeError:      # tqdm.pandas()가 선언 안된 경우
        result = article_series.map(lambda x: okt.nouns(x))

    return result


def remove_stop_words_from_each_article(article_series):
    """ series의 각 열에서 불용어 제거

    Args:
        article_series: 한 줄에 문서 하나씩

    Returns:
        series: 한 줄에 불용어 제거된 문서 하나씩

    """

    stop_words_set = set(_get_stop_words())

    try:
        result = article_series.progress_map(lambda x: [word for word in x if word not in stop_words_set])
    except AttributeError:      # tqdm.pandas()가 선언 안된 경우
        result = article_series.map(lambda x: [word for word in x if word not in stop_words_set])

    return result


def remove_one_character_from_each_article(article_series) -> pd.Series:
    """ series의 각 열에서 한 글자인 단어 제거

    Args:
        article_series: 한 줄에 문서 하나씩

    Returns:
        series: 한 줄에 한 글자 단어가 제거된 문서 하나씩

    """

    try:
        result = article_series.progress_map(lambda line: [word for word in line if len(word) > 1])
    except AttributeError:      # tqdm.pandas()가 선언 안된 경우
        result = article_series.map(lambda line: [word for word in line if len(word) > 1])

    return result


def remove_low_count_word(tokenized_article_series, min_word_count: int = 20) -> pd.Series:

    word_count_series = tokenized_article_series.explode().value_counts()
    word_to_delete = word_count_series[word_count_series <= min_word_count].index.tolist()

    return pd.Series([[i for i in article if i not in word_to_delete] for article in tokenized_article_series])


def noun(xlsx_name: str = 'test.xlsx', sheet_name=None, column_name: str = 'article', min_word_count: int = 50):
    """

    Args:
        xlsx_name:
        sheet_name:
        column_name:
        min_word_count:

    Returns:
        series: 명사 추출 -> 불용어 제거 -> 한글자 제거 -> 적게 등장한 글자 제거

    """
    if sheet_name is None:
        sheet_name = 0

    article_series = openxlsx.load_series_from_xlsx(xlsx_name, column_name, sheet_name)
    tokenized_article_series = extract_noun_for_each_article(article_series)
    tokenized_article_series = remove_stop_words_from_each_article(tokenized_article_series)
    tokenized_article_series = remove_one_character_from_each_article(tokenized_article_series)
    tokenized_article_series = remove_low_count_word(tokenized_article_series, min_word_count)

    return tokenized_article_series


def main():
    # setting
    tqdm.pandas()
    input_data, output_data = _setting()

    # preprocess - Noun
    tokenized_article_series = noun(input_data['xlsx_name'],
                                    input_data['sheet_name'],
                                    input_data['column_name'],
                                    output_data['min_word_count'])

    # save result
    openxlsx.save_to_xlsx(tokenized_article_series,
                          output_data['result_xlsx_name'],
                          output_data['result_column_name'],
                          output_data['result_sheet_name'])

    print("전처리 완료")


if __name__ == '__main__':
    main()
