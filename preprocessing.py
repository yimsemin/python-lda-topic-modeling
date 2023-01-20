from tqdm import tqdm

from konlpy.tag import Okt

import util.openxlsx as openxlsx
from stopwords.stopwordlist import get_stop_words as _get_stop_words


def _setting():
    xlsx_name = "input/12article.xlsx"
    column_name = "article"
    save_result_to = "input/12article.xlsx"

    return xlsx_name, column_name, save_result_to


def extract_noun_from_each_article(article_series):
    print("명사만을 추출합니다.")

    okt = Okt()

    try:
        result = article_series.progress_map(lambda x: okt.nouns(x))
    except AttributeError:      # tqdm.pandas()가 선언 안된 경우
        result = article_series.map(lambda x: okt.nouns(x))

    return result


def remove_stop_words_from_each_article(article_series):
    print("사전을 기준으로 불용어를 제거합니다.")

    stop_words_set = set(_get_stop_words())

    try:
        result = article_series.progress_map(lambda x: [word for word in x if word not in stop_words_set])
        # result = article_series.apply(lambda x: [word for word in x if word not in stop_words_set])
    except AttributeError:      # tqdm.pandas()가 선언 안된 경우
        result = article_series.map(lambda x: [word for word in x if word not in stop_words_set])

    return result


def remove_one_character_from_each_article(article_series):
    print("한 글자 단어를 제거합니다.")

    try:
        result = article_series.progress_map(lambda line: [word for word in line if len(word) > 1])
    except AttributeError:      # tqdm.pandas()가 선언 안된 경우
        result = article_series.map(lambda line: [word for word in line if len(word) > 1])

    return result


def noun(xlsx_name: str = 'test.xlsx', column_name: str = 'article'):
    print(">>전처리 작업: 명사 추출")

    article_series = openxlsx.load_series_from_xlsx(xlsx_name, column_name, sheet_name=0)
    article_series = extract_noun_from_each_article(article_series)
    article_series = remove_one_character_from_each_article(article_series)
    article_series = remove_stop_words_from_each_article(article_series)

    print(">>전처리 작업 완료")

    return article_series


def save_preprocessed_result(preprocessed_result,
                             save_result_to: str,
                             column_name,
                             sheet_name: str = 'preprocessed_result'):
    openxlsx.save_to_xlsx(preprocessed_result, save_result_to, column_name, sheet_name)


def main():
    print("전처리 작업을 시작합니다.")

    tqdm.pandas()

    xlsx_name, column_name, save_result_to = _setting()

    preprocessed_article_series = noun(xlsx_name, column_name)
    save_preprocessed_result(preprocessed_article_series, save_result_to, column_name)

    print("전처리 완료")


if __name__ == '__main__':
    main()
