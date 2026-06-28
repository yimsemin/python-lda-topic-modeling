""" 최적의 토픽 갯수 k를 찾기 위해 LDA 모델들의 혼란도(perplexity)와 응집도(coherence)를 조사함
"""
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from tqdm import tqdm

import lda
import util.recorder as recorder
import util.token_parser as token_parser


def _setting():
    setting = {
        'xlsx_name': 'test/input/data.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',
        'result_dir': 'test/output/',
        'result_model_dir': 'test/output/model/',
        'save_explore_html': True,
        'save_explore_topic_detail': True,
        'reuse_saved_corpus': True,
        'reuse_saved_model': True,
        'topic_number_start': 2,
        'topic_number_end': 40,
        'topic_number_interval': 1,
        'iterations': 50,
        'random_state': 4190
    }

    tokenized_article_series = _read_tokenized_article_series(setting)

    _ensure_topic_number_list(setting)

    return setting, tokenized_article_series


def _read_tokenized_article_series(setting):
    excel_data = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name'])[setting['column_name']]
    return token_parser.parse_tokenized_series(excel_data)


def _ensure_topic_number_list(setting):
    if 'topic_number_list' not in setting:
        if setting['topic_number_interval'] == 0:
            raise ValueError('topic_number_interval은 0일 수 없습니다.')
        setting['topic_number_list'] = list(range(setting['topic_number_start'],
                                                  setting['topic_number_end'] + 1,
                                                  setting['topic_number_interval']))

    if not setting['topic_number_list']:
        raise ValueError('조사할 토픽 갯수 범위가 비어 있습니다. start, end, interval 설정을 확인하세요.')


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
              x_label_name: str = 'x', y_label_name: str = 'y', save_graph_to: str = 'none',
              x_values=None):
    x_range = x_values if x_values is not None else range(range_start, range_end + 1)
    plt.plot(x_range, plot_body)
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.tight_layout()

    if save_graph_to is None or save_graph_to == 'none':
        plt.show()
    else:
        recorder.ensure_parent_dir(save_graph_to)
        plt.savefig(save_graph_to)

    plt.clf()


def get_perplexity_and_coherence_value_list(tokenized_article_series, corpus, dictionary,
                                            topic_number_list,
                                            iterations: int = 100,
                                            random_state: int = 4190,
                                            result_dir: str = 'test/output/',
                                            model_dir: str = 'test/output/model/',
                                            save_html: bool = True,
                                            save_topic_detail: bool = True,
                                            reuse_saved_model: bool = True) -> pd.DataFrame:
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
        save_html: 계산과정에서 도출된 LDA 토픽 시각화(html) 저장 여부
        save_topic_detail: 토픽별 키워드 csv와 문서별 토픽분포 csv 저장 여부
        reuse_saved_model: 기존에 만든 같은 토픽수/랜덤값 모델 재사용 여부

    Returns:
        (pd.DataFrame) 토픽 갯수 별 perplexity 및 coherence 값
    """
    values_dict = {}
    tokenized_article_series = pd.Series(tokenized_article_series).map(token_parser.parse_tokenized_article)
    recorder.ensure_dir(result_dir)
    recorder.ensure_dir(model_dir)

    created_model_names = []
    reused_model_names = []
    renamed_model_names = []
    html_skipped_messages = []
    html_note_messages = []
    for i in tqdm(topic_number_list, desc='토픽 갯수 탐색', unit='model'):
        model_name = lda.get_lda_model_name(i, random_state)
        model_path = os.path.join(model_dir, model_name)
        html_path = os.path.join(result_dir, f'{model_name}.html')
        new_model_created = False
        if reuse_saved_model:
            try:
                lda_model, loaded_model_name = lda.load_lda_model(model_dir, i, random_state)
                reused_model_names.append(model_name)
                if loaded_model_name != model_name:
                    lda_model.save(model_path)
                    renamed_model_names.append(model_name)
            except FileNotFoundError:
                lda_model = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary,
                                     passes=20, iterations=iterations, random_state=random_state)
                lda_model.save(model_path)
                new_model_created = True
                created_model_names.append(model_name)
        else:
            lda_model = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary,
                                 passes=20, iterations=iterations, random_state=random_state)
            lda_model.save(model_path)
            new_model_created = True
            created_model_names.append(model_name)

        values_dict[f'topic{i}'] = (get_perplexity(lda_model, corpus),
                                    get_coherence(lda_model, tokenized_article_series, dictionary))

        html_missing_or_empty = not os.path.exists(html_path) or os.path.getsize(html_path) == 0
        if save_html and (new_model_created or html_missing_or_empty):
            try:
                lda.save_lda_html(lda_model, corpus, dictionary, html_path, html_note_messages)
            except Exception as e:
                html_skipped_messages.append(f'{model_name}: {type(e).__name__}: {e}')

        if save_topic_detail:
            lda.save_topics_csv(lda_model, i, os.path.join(result_dir, f'{model_name}.csv'))
            topic_table = lda.get_topic_distribution_for_each_doc(lda_model, corpus)
            topic_table_path = os.path.join(result_dir, f'{model_name}_topic_table.csv')
            recorder.ensure_parent_dir(topic_table_path)
            topic_table.to_csv(topic_table_path, mode='w', encoding='utf-8')

    print(f'-- 토픽 갯수 탐색 완료: 새 모델 {len(created_model_names)}개, 기존 모델 {len(reused_model_names)}개')
    if renamed_model_names:
        print('-- 모델 파일명 정리: '+', '.join(renamed_model_names))
    if html_skipped_messages:
        print('-- HTML 저장 생략:')
        for message in html_skipped_messages:
            print(f'---- {message}')
    if html_note_messages:
        print('-- HTML 저장 특이사항:')
        for message in html_note_messages:
            print(f'---- {message}')

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
    if setting is None:
        setting, default_tokenized_article_series = _setting()
        print('-- _setting() 기본 설정값을 사용합니다.')
        if tokenized_article_series is None:
            print('-- _setting() 기본 입력 데이터를 사용합니다.')
            tokenized_article_series = default_tokenized_article_series
    elif tokenized_article_series is None:
        print('-- 전달된 setting의 입력 파일에서 LDA 토픽 갯수 탐색 입력 데이터를 읽습니다.')
        tokenized_article_series = _read_tokenized_article_series(setting)

    _ensure_topic_number_list(setting)

    reuse_saved_corpus = setting.get('reuse_saved_corpus', True)
    reuse_saved_model = setting.get('reuse_saved_model', True)
    if not reuse_saved_corpus and reuse_saved_model:
        print('-- corpus/dictionary를 새로 생성하므로 기존 LDA 모델도 재사용하지 않습니다.')
        reuse_saved_model = False

    corpus, dictionary = lda.get_corpus_and_dictionary(tokenized_article_series, setting['result_dir'],
                                                      reuse_saved_corpus)

    # LDA modeling + save result
    values_df = get_perplexity_and_coherence_value_list(tokenized_article_series, corpus, dictionary,
                                                        setting['topic_number_list'],
                                                        setting['iterations'],
                                                        setting['random_state'],
                                                        setting['result_dir'],
                                                        setting['result_model_dir'],
                                                        setting.get('save_explore_html', True),
                                                        setting.get('save_explore_topic_detail', True),
                                                        reuse_saved_model)

    # save_to_csv
    explore_csv_path = os.path.join(setting['result_dir'], 'lda__explore_topic_number.csv')
    recorder.ensure_parent_dir(explore_csv_path)
    values_df.to_csv(explore_csv_path, mode='w', encoding='utf-8',
                     header=['Perplexity', 'Coherence'], index_label='topic number')

    # save_to_graph
    perplexity_list = values_df['perplexity'].tolist()
    draw_plot(perplexity_list, setting['topic_number_list'][0], setting['topic_number_list'][-1],
              'Number of topics', 'Perplexity', os.path.join(setting['result_dir'], 'lda__perplexity_value.png'),
              setting['topic_number_list'])

    coherence_list = values_df['coherence'].tolist()
    draw_plot(coherence_list, setting['topic_number_list'][0], setting['topic_number_list'][-1],
              'Number of topics', 'Coherence', os.path.join(setting['result_dir'], 'lda__coherence_value.png'),
              setting['topic_number_list'])


def main():
    with recorder.WithTimeRecorder('lda_explore_topic_number'):
        lda_explore_topic_number()


if __name__ == '__main__':
    main()
