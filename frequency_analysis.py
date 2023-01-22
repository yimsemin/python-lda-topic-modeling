import pandas as pd
from tqdm import tqdm

import util.openxlsx as openxlsx
import util.recorder as recorder


def _setting():
    # input
    input_data = {
        'xlsx_name': 'input/test.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',
    }

    # output
    output_data = {
        'result_xlsx_name': 'result/test_frequency_analysis.xlsx',
        'result_sheet_name': 'frequency_analysis',
        'result_csv_name': 'result/test_frequency_analysis.csv',
        'result_save_type': 'csv',       # xlsx or csv
        'min_word_count': 50
    }

    tokenized_article_series = openxlsx.load_series_from_xlsx(input_data['xlsx_name'],
                                                              input_data['column_name'],
                                                              input_data['sheet_name'],
                                                              is_list_in_list=True)

    return input_data, output_data, tokenized_article_series


def frequency_analysis(tokenized_article_series, min_word_count: int = 20) -> pd.Series:
    """ 단어 및 빈도수를 내림차순으로 반환

    Args:
        tokenized_article_series:
        min_word_count:

    Returns:
        pd.Series : 각 열마다 단어와 빈도수

    """
    word_count_series = tokenized_article_series.explode().value_counts(ascending=False).rename('word_count')

    # min_word_count 보다 높은 단어만 제시
    return word_count_series[word_count_series >= min_word_count]


def frequency_analysis_by_time_slice():
    # TODO time_slice로 나눈 것 기준으로 빈도수 분석
    pass


def main():
    # setting
    tqdm.pandas()
    _, output_data, tokenized_word_series = _setting()

    # frequency analysis
    with recorder.WithTimeRecorder('빈도 분석'):
        frequency_result = frequency_analysis(tokenized_word_series, output_data['min_word_count'])

    # save result
    if output_data['result_save_type'] == 'xlsx':
        with pd.ExcelWriter(output_data['result_xlsx_name'], mode='w', engine='openpyxl') as writer:
            frequency_result.to_excel(writer, sheet_name=output_data['result_sheet_name'])

    elif output_data['result_save_type'] == 'csv':
        frequency_result.to_csv(output_data['result_csv_name'], mode='w', encoding='utf-8',
                                header=['count'], index_label='word')

    else:
        raise ValueError('결과 저장은 xlsx 또는 csv으로 설정해주세요.')


if __name__ == '__main__':
    main()
