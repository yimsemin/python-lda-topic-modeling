""" 각 행에 전처리된 문서가 기록된 엑셀파일에서, 전체 문서에 대해 단어 빈도분석 실시 -> 결과를 csv로 저장
"""
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib'))

import pandas as pd
from wordcloud import WordCloud

import util.recorder as recorder


def _split_tokenized_article(line):
    if isinstance(line, list):
        return line
    if line is None or pd.isna(line):
        return []

    return [word for word in str(line).split(',') if word]


def _get_available_word_cloud_font(my_font: str = 'font/NanumGothic.ttf'):
    if my_font is not None and os.path.isfile(my_font):
        return my_font

    if my_font is not None:
        print(f'-- 워드클라우드용 폰트({my_font})가 없음')

    # 설정 폰트가 없을 때 한글이 네모로 깨지지 않도록 흔한 시스템 폰트를 사용한다.
    fallback_fonts = [
        '/System/Library/Fonts/AppleSDGothicNeo.ttc',
        '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
        '/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf',
        'C:/Windows/Fonts/malgun.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
    ]
    for font_path in fallback_fonts:
        if os.path.isfile(font_path):
            print(f'-- 대체 한글 폰트({font_path})를 사용합니다.')
            return font_path

    print('-- 대체 한글 폰트를 찾지 못해 기본 폰트로 워드클라우드를 생성합니다.')
    return None


def _setting():
    setting = {
        # input
        'xlsx_name': 'test/input/data.xlsx',
        'sheet_name': 'preprocessed',                                   # 시트 이름 str 입력
        'column_name': 'article',                                       # 전처리를 한 문서가 있는 열의 첫번째 행 이름 str 입력
        # 1번째 행      article (제목 줄)
        # 2번째 행      키워드,키워드,키워드,키워드 ...
        # 3번째 행      키워드,키워드,키워드,키워드 ...
        # 4번째 행      키워드,키워드,키워드,키워드 ...
        # ...

        # output
        'result_csv_name': 'test/output/frequency_analysis.csv',        # 파일이 이미 존재하면 덮어씀
        'result_word_cloud_name': 'test/output/word_cloud.png',         # 파일이 이미 존재하면 덮어씀
        'word_cloud_font': 'font/NanumGothic.ttf',                      # 워드클라우드용 폰트
        'min_word_count': 50                                            # n회 이하 나타난 단어는 결과에서 제거
    }

    excel_data = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name'])[setting['column_name']]
    tokenized_article_series = excel_data.map(_split_tokenized_article)
    # 0      [키워드, 키워드, 키워드 ...
    # 1      [키워드, 키워드, 키워드 ...
    # 2      [키워드, 키워드, 키워드 ...
    # ...
    # Name: article, Length: 000, dtype: object

    return setting, tokenized_article_series


def count_frequency(tokenized_article_series: pd.Series, min_word_count: int = 50) -> pd.Series:
    """ 단어(토큰)와 빈도수를 내림차순으로 반환

    Args:
        tokenized_article_series(pd.Series): 각 줄은 토큰으로 구성된 리스트 예: [키워드, 키워드, 키워드 ... ]
        min_word_count: 적게 등장한 단어를 결과에서 제거할 때 그 기준

    Returns:
        (pd.Series) 각 열마다 단어와 빈도수
    """
    words = tokenized_article_series.explode().dropna()
    word_count_series = words[words != ''].value_counts(ascending=False).rename('word_count')

    # min_word_count 보다 많이 등장한 단어만 제시
    return word_count_series[word_count_series > min_word_count]


def word_cloud_analysis(csv_location: str, save_graph_to: str = None, my_font: str = 'font/NanumGothic.ttf'):
    """ frequency_analysis() 결과로 저장된 csv 파일을 읽어와서 워드클라우드 작성
    """
    df = pd.read_csv(csv_location, index_col=0)
    if df.empty:
        print('-- 빈도분석 결과가 비어 있어 워드클라우드를 생성하지 않습니다.')
        return

    my_dict = dict(zip(df.index, [x for y in df.to_numpy().tolist() for x in y]))
    my_dict = {word: count for word, count in my_dict.items() if pd.notna(word) and pd.notna(count) and count > 0}
    if not my_dict:
        print('-- 워드클라우드에 사용할 단어가 없어 생성하지 않습니다.')
        return

    # 폰트 확인
    my_font = _get_available_word_cloud_font(my_font)

    # wordcloud에 대한 세부설정은 아래 웹사이트 참조
    # https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
    wordcloud = WordCloud(font_path=my_font, width=3200, height=1600, background_color='white').fit_words(my_dict)

    if save_graph_to is None or save_graph_to == 'none':
        import matplotlib.pyplot as plt
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.clf()
    else:
        recorder.ensure_parent_dir(save_graph_to)
        wordcloud.to_file(save_graph_to)


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
    if setting is None:
        setting, default_tokenized_article_series = _setting()
        if tokenized_article_series is None:
            tokenized_article_series = default_tokenized_article_series
    elif tokenized_article_series is None:
        excel_data = pd.read_excel(setting['xlsx_name'],
                                   sheet_name=setting['sheet_name'])[setting['column_name']]
        tokenized_article_series = excel_data.map(_split_tokenized_article)

    # frequency analysis
    frequency_result = count_frequency(tokenized_article_series, setting['min_word_count'])

    # save result
    recorder.ensure_parent_dir(setting['result_csv_name'])
    frequency_result.to_csv(setting['result_csv_name'], mode='w', encoding='utf-8',
                            header=['count'], index_label='word')

    # word cloud
    word_cloud_analysis(setting['result_csv_name'], setting['result_word_cloud_name'], setting['word_cloud_font'])


def main():
    with recorder.WithTimeRecorder('빈도분석'):
        frequency_analysis()


if __name__ == '__main__':
    main()
