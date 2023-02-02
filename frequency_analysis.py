""" 전처리된 문서들에 대하여 단어의 빈도분석 실시 후 csv로 저장
"""
import pandas as pd

import util.openxlsx as openxlsx
import util.recorder as recorder


def _setting():
    setting = {
        # input
        'xlsx_name': 'input/test.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',

        # output
        'result_csv_name': 'result/test_frequency_analysis.csv',
        'min_word_count': 50
    }

    tokenized_article_series = openxlsx.load_series_from_xlsx(setting['xlsx_name'],
                                                              setting['column_name'],
                                                              setting['sheet_name'],
                                                              is_list_in_list=True)

    return setting, tokenized_article_series


def count_frequency(tokenized_article_series: pd.Series, min_word_count: int = 20) -> pd.Series:
    """ 단어 및 빈도수를 내림차순으로 반환

    Args:
        tokenized_article_series(pd.Series): 한 줄에 토큰화된 문서 하나씩
        min_word_count: 적게 등장한 단어를 결과에서 제거할 때 그 기준

    Returns:
        (pd.Series) 각 열마다 단어와 빈도수
    """
    word_count_series = tokenized_article_series.explode().value_counts(ascending=False).rename('word_count')

    # min_word_count 보다 높은 단어만 제시
    return word_count_series[word_count_series >= min_word_count]


def frequency_analysis_by_time_slice():
    # TODO time_slice로 나눈 것 기준으로 빈도수 분석
    pass


def frequency_analysis(setting: dict = None, tokenized_article_series: pd.Series = None):
    """ 전처리된 문서들에 대해 빈도분석을 하여 csv로 저장

    Args:
        setting: 설정값 불러오기
            setting['result_csv_name']: str = 빈도분석 결과를 저장할 csv파일, 예: 'where/filename.csv'
            setting['min_word_count']: int = 적게 등장한 단어를 결과에서 표시하지 않을 때 그 기준
        tokenized_article_series: 한 줄에 토큰화된 문서 하나씩
    """
    if setting or tokenized_article_series is None:
        setting, tokenized_article_series = _setting()

    # frequency analysis
    frequency_result = count_frequency(tokenized_article_series, setting['min_word_count'])

    # save result
    frequency_result.to_csv(setting['result_csv_name'], mode='w', encoding='utf-8',
                            header=['count'], index_label='word')


def main():
    with recorder.WithTimeRecorder('빈도분석'):
        frequency_analysis()


if __name__ == '__main__':
    main()
