""" 각 행에 1개의 문서가 기록된 엑셀파일에서 각 행(문서)을 전처리 -> 새로운 시트에 전처리 결과를 저장
"""
import pandas as pd
from tqdm import tqdm

from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

import util.recorder as recorder
import util.token_parser as token_parser


def _setting():
    setting = {
        # input - 전처리를 수행할 엑셀파일
        'xlsx_name': 'test/input/data.xlsx',
        'sheet_name': 0,                                # 시트 이름 str 입력 / 0 입력 -> 가장 왼쪽에 있는 시트를 선택
        'column_name': 'article',                       # 전처리 대상 문서가 있는 열의 첫번째 행 이름 str 입력
        # 1번째 행     article (제목 줄)
        # 2번째 행     1줄에 1개의 문서 ...
        # 3번째 행     1줄에 1개의 문서 ...
        # 4번째 행     1줄에 1개의 문서 ...
        # ...

        'stopwordlist_location': 'test/input/stopwordlist.txt',  # 불용어 사전 위치

        # output - 전처리 결과에 대한 설정
        'result_sheet_name': 'preprocessed',            # 결과를 저장할 시트 이름 / 시트가 이미 존재하면 덮어씀
        'min_word_count': 50                            # n회 이하 나타난 단어는 삭제함
    }

    article_series = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name'])[setting['column_name']]

    return setting, article_series


def load_stopwords(stopwordlist_location: str = 'test/input/stopwordlist.txt') -> Stopwords:
    """ kiwipiepy 기본 불용어에 사용자 정의 불용어를 추가 """
    stopwords = Stopwords()

    try:
        with open(stopwordlist_location, 'r', encoding='utf-8') as f:
            print('-- 저장된 불용어 사전을 불러옵니다.')
            txt_lines = f.read().splitlines()
    except FileNotFoundError:
        print('-- 사용자 정의 불용어 사전이 없어 kiwipiepy 기본 불용어만 사용합니다.')
        return stopwords

    custom_stopwords = []
    comments = [line for line in txt_lines if '#' in line]
    if not comments:
        print('---- 주석 없음')
    else:
        for i in comments:
            print('---- '+str(i))

    for raw_line in txt_lines:
        line = raw_line.strip()
        if not line or '#' in line:
            continue

        if '/' in line:
            form, tag = line.rsplit('/', 1)
            custom_stopwords.append((form.strip(), tag.strip()))
        else:
            custom_stopwords.append((line, 'NNG'))

    if custom_stopwords:
        stopwords.add(custom_stopwords)
        print('-- 사용자 정의 불용어 예시 : '+', '.join([f'{form}/{tag}' for form, tag in custom_stopwords[0:4]])+' ...')

    return stopwords


def extract_noun_from_each_article(article_series: pd.Series, stopwords: Stopwords = None) -> pd.Series:
    """ 각 열의 문서에 대해 kiwipiepy 기반으로 명사만 추출

    Args:
        article_series(pd.Series): 한 줄에 문서 하나씩

    Returns:
        (pd.Series) 한 줄에 명사만 추출된(토큰화된) 문서 하나씩
    """
    tqdm.pandas()
    kiwi = Kiwi()
    target_tags = {'NNG', 'NNP'}

    def extract_noun(article):
        # 빈 셀은 pandas에서 NaN으로 읽히므로 빈 문서로 처리한다.
        if article is None or pd.isna(article):
            return []

        tokens = kiwi.tokenize(str(article))
        if stopwords is not None:
            tokens = stopwords.filter(tokens)

        return [token.form for token in tokens if token.tag in target_tags]

    return article_series.progress_map(extract_noun)


def remove_stop_words_from_each_article(tokenized_article_series: pd.Series,
                                        stopwordlist_location: str = 'test/input/stopwordlist.txt') -> pd.Series:
    """ 이미 명사로 추출된 문서 묶음에서 특정 키워드를 일괄 삭제하는 레거시 보조 함수

    기본 전처리 파이프라인에서는 사용하지 않음.
    제거할 키워드 사전은 1줄에 1개씩 작성, 단어/품사 형식을 쓰면 단어만 사용, #이 포함된 줄은 주석으로 처리함

    Args:
        tokenized_article_series(pd.Series): 각 줄은 토큰으로 구성된 리스트 예: [키워드, 키워드, 키워드 ... ]
        stopwordlist_location(str): 제거할 키워드 사전(txt) 위치

    Returns:
        (pd.Series) 한 줄에 특정 키워드가 제거된 문서 하나씩
    """
    tqdm.pandas()

    # 함수 이름의 stop_words는 과거 호환을 위해 유지한다.
    try:
        with open(stopwordlist_location, 'r', encoding='utf-8') as f:
            print('-- 저장된 제거할 키워드 사전을 불러옵니다.')
            txt_lines = f.read().splitlines()

        comments = [line for line in txt_lines if '#' in line]
        if not comments:
            print('---- 주석 없음')
        else:
            for i in comments:
                print('---- '+str(i))       # 제거할 키워드 사전의 코멘트 출력

        remove_word_list = []
        for raw_line in txt_lines:
            line = raw_line.strip()
            if not line or '#' in line:
                continue

            if '/' in line:
                remove_word_list.append(line.rsplit('/', 1)[0].strip())
            else:
                remove_word_list.append(line)

    except FileNotFoundError:
        print('-- 제거할 키워드 사전을 찾지 못했습니다. 키워드 제거를 건너뜁니다.')
        return tokenized_article_series

    if not remove_word_list:
        print('-- 제거할 키워드가 없습니다. 키워드 제거를 건너뜁니다.')
        return tokenized_article_series

    print('-- 제거할 키워드 사전 예시 : '+', '.join(remove_word_list[0:4])+' ...')

    remove_words_set = set(remove_word_list)

    return tokenized_article_series.progress_map(lambda x: [word for word in x if word not in remove_words_set])


def remove_one_character_from_each_article(tokenized_article_series) -> pd.Series:
    """ 각 열의 문서에 대해 한 글자인 단어를 제거하는 레거시 보조 함수

    kiwipiepy 기반 기본 전처리 파이프라인에서는 사용하지 않음.

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
                         index=tokenized_article_series.index,
                         name=tokenized_article_series.name)


def preprocessing_noun(setting: dict = None, article_series: pd.Series = None):
    """ pd.Series 데이터를 전처리하여 xlsx 파일에 저장

    수행하는 전처리: 명사 추출 및 불용어 제거 -> 적게 등장한 단어 제거

    Args:
        setting: 설정값 불러오기
        article_series: 한 줄에 문서 하나씩
    """
    if setting is None:
        setting, default_article_series = _setting()
        if article_series is None:
            article_series = default_article_series
    elif article_series is None:
        article_series = pd.read_excel(setting['xlsx_name'],
                                       sheet_name=setting['sheet_name'])[setting['column_name']]

    stopwords = load_stopwords(setting['stopwordlist_location'])

    # preprocess - Noun
    print('1단계: 명사를 추출하고 불용어를 제거합니다.')
    tokenized_article_series = extract_noun_from_each_article(article_series, stopwords)
    print('2단계: 적게 등장한 단어를 제거합니다.')
    tokenized_article_series = remove_low_count_word(tokenized_article_series, setting['min_word_count'])
    # 0      [키워드, 키워드, 키워드 ...
    # 1      [키워드, 키워드, 키워드 ...
    # 2      [키워드, 키워드, 키워드 ...
    # ...
    # Name: article, Length: 000, dtype: object

    token_parser.log_empty_documents(
        tokenized_article_series,
        '전처리 결과',
        '-- 안내: 빈 문서는 삭제하지 않고 저장합니다. 이후 LDA 계열 분석에서 의미 없는 토픽 분포로 반영될 수 있습니다.',
        source_series=article_series
    )
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
