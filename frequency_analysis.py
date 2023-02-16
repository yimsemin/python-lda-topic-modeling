""" 전처리된 문서들에 대하여 단어의 빈도분석 실시 후 csv로 저장
"""
import pandas as pd

import util.recorder as recorder


def _setting():
    setting = {
        # input
        'xlsx_name': 'input/test.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',
        # article         <- 제목 줄
        # 키워드,키워드,키워드,키워드 ...
        # 키워드,키워드,키워드,키워드 ...
        # 키워드,키워드,키워드,키워드 ...
        # 키워드,키워드,키워드,키워드 ...

        # output -- 폴더는 미리 만들어둬야 함
        'result_csv_name': 'result/test/frequency_analysis.csv',        # 파일이 이미 존재하면 덮어씀
        'min_word_count': 50                                            # n회 이하 나타난 단어는 결과에서 제거
    }

    excel_data = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name'])[setting['column_name']]
    tokenized_article_series = excel_data.map(lambda line: line.split(','), na_action='ignore')
    # 0      [키워드, 키워드, 키워드 ...
    # 1      [키워드, 키워드, 키워드 ...
    # 2      [키워드, 키워드, 키워드 ...
    # ...
    # Name: article, Length: 000, dtype: object

    return setting, tokenized_article_series


def count_frequency(tokenized_article_series: pd.Series, min_word_count: int = 20) -> pd.Series:
    """ 단어 및 빈도수를 내림차순으로 반환

    Args:
        tokenized_article_series(pd.Series): 각 줄은 토큰으로 구성된 리스트 예: [키워드, 키워드, 키워드 ... ]
        min_word_count: 적게 등장한 단어를 결과에서 제거할 때 그 기준

    Returns:
        (pd.Series) 각 열마다 단어와 빈도수
    """
    word_count_series = tokenized_article_series.explode().value_counts(ascending=False).rename('word_count')

    # min_word_count 보다 높은 단어만 제시
    return word_count_series[word_count_series >= min_word_count]


def word_cloud_analysis():
    # TODO 워드클라우드 만들기
    pass


def frequency_analysis_by_group():
    # TODO 그룹으로 나눈 것(예: time_slice) 기준으로 빈도수 분석
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
