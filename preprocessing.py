""" 각 행에 1개의 문서가 기록된 엑셀파일에서 각 행(문서)을 전처리 -> 새로운 시트에 전처리 결과를 저장
"""
import pandas as pd
from tqdm import tqdm

from konlpy.tag import Okt

import util.recorder as recorder


def _setting():
    setting = {
        # input - 전처리를 수행할 엑셀파일
        'xlsx_name': 'test/test.xlsx',
        'sheet_name': 0,                                # 시트 이름 str 입력 / 0 입력 -> 가장 왼쪽에 있는 시트를 선택
        'column_name': 'article',                       # 전처리 대상 문서가 있는 열의 첫번째 행 이름 str 입력
        # 1번째 행     article (제목 줄)
        # 2번째 행     1줄에 1개의 문서 ...
        # 3번째 행     1줄에 1개의 문서 ...
        # 4번째 행     1줄에 1개의 문서 ...
        # ...

        'stopwordlist_location': 'stopwords/stopwordlist.txt',   # 불용어 사전 위치

        # output - 전처리 결과에 대한 설정
        'result_sheet_name': 'preprocessed',            # 결과를 저장할 시트 이름 / 시트가 이미 존재하면 덮어씀
        'min_word_count': 50                            # n회 이하 나타난 단어는 삭제함
    }

    article_series = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name'])[setting['column_name']]

    return setting, article_series


def extract_noun_from_each_article(article_series: pd.Series) -> pd.Series:
    """ 각 열의 문서에 대해 okt(Open Korean Text) 기반으로 명사만 추출

    Args:
        article_series(pd.Series): 한 줄에 문서 하나씩

    Returns:
        (pd.Series) 한 줄에 명사만 추출된(토큰화된) 문서 하나씩
    """
    tqdm.pandas()
    okt = Okt()

    return article_series.progress_map(lambda x: okt.nouns(x))


def remove_stop_words_from_each_article(tokenized_article_series: pd.Series,
                                        stopwordlist_location: str = 'stopwords/stopwordlist.txt') -> pd.Series:
    """ 각 열의 문서에 대해 불용어 사전 기준으로 불용어 제거
    불용어 사전은 1줄에 1단어씩 작성, #이 포함된 단어는 주석으로 처리함

    Args:
        tokenized_article_series(pd.Series): 각 줄은 토큰으로 구성된 리스트 예: [키워드, 키워드, 키워드 ... ]
        stopwordlist_location(str): 불용어 사전(txt) 위치

    Returns:
        (pd.Series) 한 줄에 불용어 제거된 문서 하나씩
    """
    tqdm.pandas()

    # 불용어 사전 불러오기
    try:
        with open(stopwordlist_location, 'r') as f:
            print('-- 저장된 불용어 사전을 불러옵니다.')
            txt_lines = f.read().splitlines()

        comments = [line for line in txt_lines if '#' in line]
        if not comments:
            print('---- 주석 없음')
        else:
            [print('---- '+str(i)) for i in comments]       # 불용어 사전의 코멘트 출력

        stop_word_list = [line for line in txt_lines if '#' not in line]

    except FileNotFoundError:
        print('-- 기본 불용어 사전을 사용합니다.')
        stop_word_list = ['\n', '을', '를', '은', '가', '는', '이', '도', '수', '며', '고']      # 기본 불용어

    print('-- 불용어 사전 예시 : '+', '.join(stop_word_list[0:4])+' ...')

    stop_words_set = set(stop_word_list)

    return tokenized_article_series.progress_map(lambda x: [word for word in x if word not in stop_words_set])


def remove_one_character_from_each_article(tokenized_article_series) -> pd.Series:
    """ 각 열의 문서에 대해 한 글자인 단어 제거

    Args:
        tokenized_article_series(pd.Series): 각 줄은 토큰으로 구성된 리스트 (예: [키워드, 키워드, 키워드 ... ])

    Returns:
        (pd.Series) 한 줄에 한 글자 단어가 제거된 문서 하나씩
    """
    tqdm.pandas()

    return tokenized_article_series.progress_map(lambda line: [word for word in line if len(word) > 1])


def remove_low_count_word(tokenized_article_series, min_word_count: int = 50) -> pd.Series:
    """ 각 열의 문서에 대해 적게 등장한 단어 제거

    Args:
        tokenized_article_series(pd.Series): 각 줄은 토큰으로 구성된 리스트 예: [키워드, 키워드, 키워드 ... ]
        min_word_count: 적게 등장한 단어를 제거할 때 그 기준, 0 또는 음수 입력시 진행하지 않음

    Returns:
        (pd.Series) 한 줄에 적게 등장한 단어가 제거된 문서 하나씩
    """
    # TODO 데이터가 많아지면 상당히 느려짐 / 멀티프로세싱이 가능할지?

    # 최소 카운트가 0일 경우 함수 생략
    if min_word_count <= 0:
        print(f'-- 적게 등장한 단어 제거의 기준이 0 이하로 입력되었습니다. 제거를 진행하지 않습니다.')
        return tokenized_article_series

    else:
        word_count_series = tokenized_article_series.explode().value_counts()
        deleted_word_count_series = word_count_series[word_count_series <= min_word_count]

        print('-- 다음의 단어들을 제거합니다. (단어 / 등장 횟수) :')
        print(deleted_word_count_series)
        # 지운 단어를 별도로 저장하고 싶을 때 활용
        # deleted_word_count_series.to_csv('save_result_to_str', mode='w', encoding='utf-8',
        #                                  header=['deleted word', 'count'])

        delete_word = set(deleted_word_count_series.index.tolist())

        return pd.Series([[i for i in article if i not in delete_word] for article in tqdm(tokenized_article_series)],
                         name=tokenized_article_series.name)


def preprocessing_noun(setting: dict = None, article_series: pd.Series = None):
    """ pd.Series 데이터를 전처리하여 xlsx 파일에 저장

    수행하는 전처리: 명사 추출 -> 불용어 제거 -> 한글자 제거 -> 적게 등장한 글자 제거

    Args:
        setting: 설정값 불러오기
        article_series: 한 줄에 문서 하나씩
    """
    if setting or article_series is None:
        setting, article_series = _setting()

    # preprocess - Noun
    print('1단계: 명사를 추출합니다.')
    tokenized_article_series = extract_noun_from_each_article(article_series)
    print('2단계: 불용어를 제거합니다.')
    tokenized_article_series = remove_stop_words_from_each_article(tokenized_article_series,
                                                                   setting['stopwordlist_location'])
    print('3단계: 한글자 단어를 제거합니다.')
    tokenized_article_series = remove_one_character_from_each_article(tokenized_article_series)
    print('4단계: 적게 등장한 단어를 제거합니다.')
    tokenized_article_series = remove_low_count_word(tokenized_article_series, setting['min_word_count'])
    # 0      [키워드, 키워드, 키워드 ...
    # 1      [키워드, 키워드, 키워드 ...
    # 2      [키워드, 키워드, 키워드 ...
    # ...
    # Name: article, Length: 000, dtype: object

    # TODO 모든 단어가 삭제되었을 때 비어버린 문서 처리
    # TODO: 'n개 이하의 문서에서만 등장한 단어 제거' 추가

    # save result
    with pd.ExcelWriter(setting['xlsx_name'], mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        # 리스트를 쉼표 기준으로 분해한 다음 저장
        data_to_save = tokenized_article_series.map(lambda word: ','.join(word))
        data_to_save.to_excel(writer, sheet_name=setting['result_sheet_name'])
        # 	article
        # 0	키워드,키워드,키워드,키워드 ...
        # 1	키워드,키워드,키워드,키워드 ...
        # 2	키워드,키워드,키워드,키워드 ...
        # 3	키워드,키워드,키워드,키워드 ...


def main():
    with recorder.WithTimeRecorder('전처리'):
        preprocessing_noun()


if __name__ == '__main__':
    main()
