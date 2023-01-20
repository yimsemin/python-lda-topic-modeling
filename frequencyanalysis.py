import sys

import pandas as pd
from tqdm import tqdm

import preprocessing
import util.openxlsx as openxlsx
import util.recorder as recorder


def _setting():
    xlsx_name = "input/6-11article.xlsx"
    column_name = "article"
    save_result_to = "result/6-11article.txt"
    is_already_preprocessed = False

    if is_already_preprocessed:
        preprocessed_article_series = openxlsx.load_series_from_xlsx(xlsx_name,
                                                                     column_name,
                                                                     sheet_name='preprocessed_result',
                                                                     is_list_in_list=True)
    else:
        preprocessed_article_series = preprocessing.noun(xlsx_name, column_name)

    return xlsx_name, column_name, save_result_to, preprocessed_article_series


def flat_list_of_lists(list_of_lists) -> pd.Series:
    if type(list_of_lists) not in [pd.Series, list]:
        raise ValueError("평탄화 작업은 Series 또는 list만 가능합니다.")

    flatten_series = pd.Series([word for line in list_of_lists for word in line])

    return flatten_series


def frequency_analysis(preprocessed_article_series, save_to_this_file: str = 'test.txt'):
    flatten_series = flat_list_of_lists(preprocessed_article_series)

    keyword_dict = dict(flatten_series.value_counts(ascending=False))

    file_type = save_to_this_file.split('.')[-1]
    if file_type == 'txt':
        with recorder.WithTxtRecorder(save_to_this_file, 'w', 'utf-8') as sys.stdout:
            print('분석 일자:', recorder.print_now())
            print('\n==[ 전     체 ]==========================================')
            print(keyword_dict)
            print("\n==[ 세로로 전체 ]==========================================")
            for key, value in keyword_dict.items():
                print(key, value, sep=" : ")
    elif file_type == 'xlsx':
        openxlsx.save_to_xlsx(keyword_dict, save_to_this_file, 'article', 'frequency_analysis_result')
    else:
        raise ValueError("결과 저장은 .xlsx 또는 .txt에만 가능합니다.")


def main():
    tqdm.pandas()

    xlsx_name, column_name, save_result_to, preprocessed_article = _setting()

    with recorder.WithTimeRecorder('빈도 분석'):
        frequency_analysis(preprocessed_article, save_result_to)


if __name__ == '__main__':
    main()
