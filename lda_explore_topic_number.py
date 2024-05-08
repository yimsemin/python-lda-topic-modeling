""" 최적의 토픽 갯수 k를 찾기 위해 LDA 모델들의 혼란도(perplexity)와 응집도(coherence)를 조사함
"""
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from tqdm import tqdm

import lda
import util.recorder as recorder


def _setting():
    setting = {
        # input
        'xlsx_name': 'test/test.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',

        # output -- 폴더는 미리 만들어둬야 함
        'result_dir': 'test/',
        'result_model_dir': 'test/model/',

        # 조사할 토픽 갯수 범위
        'topic_number_start': 2,
        'topic_number_end': 40,
        'topic_number_interval': 1,         # 시작번호부터 n씩 증가하면서 조샇함

        # LDA 모델 생성
        'iterations': 50,
        'random_state': 4190
    }

    excel_data = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name'])[setting['column_name']]
    tokenized_article_series = excel_data.map(lambda line: line.split(','), na_action='ignore')
    # 0      [키워드, 키워드, 키워드 ...
    # 1      [키워드, 키워드, 키워드 ...
    # 2      [키워드, 키워드, 키워드 ...
    # ...
    # Name: article, Length: 000, dtype: object

    setting['topic_number_list'] = list(range(setting['topic_number_start'],
                                              setting['topic_number_end'] + 1,
                                              setting['topic_number_interval']))

    return setting, tokenized_article_series


def get_perplexity(lda_model, corpus):
    """ 입력된 LDA 모델의 복잡도(perplexity)를 계산

    낮을수록 좋으나 토픽 갯수가 많아지면 낮아지는 경향을 보임
    급격하게 낮아지는 구간을 보는 것도 방법

    Args:
        lda_model: LDA 모델
        corpus: 말뭉치

    Returns:
        (float) perplexity 값
    """
    return lda_model.log_perplexity(corpus)


def get_coherence(lda_model, tokenized_article_series, dictionary):
    """ 입력된 LDA 모델의 응집도(coherence)를 계산

    높을수록 의미론적 일관성이 높아서 좋음
    일정 수준 이상으로 잘 올라가지 않음, 요동치는 구간의 시작점을 보는 듯

    Args:
        lda_model: LDA 모델
        tokenized_article_series(pd.Series): 한 줄에 토큰화된 문서 하나씩
        dictionary: 딕셔너리

    Returns:
        (float) coherence 값
    """
    coherence_model_lda = CoherenceModel(model=lda_model,
                                         texts=tokenized_article_series,
                                         dictionary=dictionary,
                                         coherence='c_v',
                                         topn=10)
    # topn (int, optional) – 토픽을 대표하는 단어들 중, 상위 n개의 중요성을 가지는 단어를 반환

    return coherence_model_lda.get_coherence()


def draw_plot(plot_body, range_start: int = 2, range_end: int = 15,
              x_label_name: str = 'x', y_label_name: str = 'y', save_graph_to: str = 'none'):
    x_range = range(range_start, range_end + 1)
    plt.plot(x_range, plot_body)
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.tight_layout()

    if save_graph_to == 'none':
        plt.show()
    else:
        plt.savefig(save_graph_to)

    plt.clf()


def get_perplexity_and_coherence_value_list(tokenized_article_series, corpus, dictionary,
                                            topic_number_list,
                                            iterations: int = 100,
                                            random_state: int = 4190,
                                            result_dir: str = 'test/',
                                            model_dir: str = 'test/model/') -> pd.DataFrame:
    """

    Args:
        tokenized_article_series: 각 줄은 토큰으로 구성된 리스트 예: [키워드, 키워드, 키워드 ... ]
        corpus: 말뭉치
        dictionary: 딕셔너리
        topic_number_list: 구하고자 하는 토픽 갯수의 리스트
        iterations: LDA 모델 계산 시 iteration
        random_state: LDA 모델 계산 시 random_state
        result_dir: 계산과정에서 도출된 LDA 토픽 시각화(html) 결과 저장 위치
        model_dir: 계산과정에서 도출된 LDA 모델 저장 위치

    Returns:
        (pd.DataFrame) 토픽 갯수 별 perplexity 및 coherence 값
    """
    values_dict = {}

    for i in tqdm(topic_number_list):
        try:
            lda_model = LdaModel.load(model_dir + f'lda_k{i}_rd{random_state}')
        except FileNotFoundError:
            print(f'>> 토픽 갯수 {i}개의 lda_model을 새로 생성합니다.')
            lda_model = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary,
                                 passes=20, iterations=iterations, random_state=random_state)
            lda_model.save(model_dir + f'lda_k{i}_rd{random_state}')
            lda.save_lda_html(lda_model, corpus, dictionary, result_dir + f'lda_k{i}_rd{random_state}.html')

        values_dict[f'topic{i}'] = (get_perplexity(lda_model, corpus),
                                    get_coherence(lda_model, tokenized_article_series, dictionary))

    values_df = pd.DataFrame.from_dict(values_dict, orient='index', columns=['perplexity', 'coherence'])
    #          perplexity  coherence
    # topic2    -4.879550   0.625420
    # topic3    -4.810820   0.616195
    # topic4    -4.769375   0.546547
    # topic5    -4.745952   0.562596
    # ...

    return values_df


def lda_explore_topic_number(setting: dict = None, tokenized_article_series: pd.Series = None):
    # setting
    if setting or tokenized_article_series is None:
        setting, tokenized_article_series = _setting()

    corpus, dictionary = lda.get_corpus_and_dictionary(tokenized_article_series, setting['result_dir'])

    # LDA modeling + save result
    values_df = get_perplexity_and_coherence_value_list(tokenized_article_series, corpus, dictionary,
                                                        setting['topic_number_list'],
                                                        setting['iterations'],
                                                        setting['random_state'],
                                                        setting['result_dir'],
                                                        setting['result_model_dir'])

    # save_to_csv
    values_df.to_csv(setting['result_dir'] + 'lda__explore_topic_number.csv', mode='w', encoding='utf-8',
                     header=['Perplexity', 'Coherence'], index_label='topic number')

    # save_to_graph
    perplexity_list = values_df['perplexity'].tolist()
    draw_plot(perplexity_list, setting['topic_number_list'][0], setting['topic_number_list'][-1],
              'Number of topics', 'Perplexity', setting['result_dir'] + 'lda__perplexity_value.png')

    coherence_list = values_df['coherence'].tolist()
    draw_plot(coherence_list, setting['topic_number_list'][0], setting['topic_number_list'][-1],
              'Number of topics', 'Coherence', setting['result_dir'] + 'lda__coherence_value.png')


def main():
    with recorder.WithTimeRecorder('lda_explore_topic_number'):
        lda_explore_topic_number()


if __name__ == '__main__':
    main()
