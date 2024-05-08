""" 전처리된 텍스트에 대한 LDA 모델링을 실시하고 시각화함
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)       # pandas의 future warning 가리기 위함

import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensim_vis
from gensim import corpora
from gensim.models import LdaModel
from tqdm import tqdm

import util.recorder as recorder


def _setting():
    # input
    setting = {
        'xlsx_name': 'test/test.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',

        # output -- 폴더는 미리 만들어둬야 함
        'result_dir': 'test/',
        'result_model_dir': 'test/model/',

        # model setting
        # 토픽의 갯수가 정해졌다면, 토픽 갯수를 고정시키고 여러 모델을 만들어 봄
        'num_topics': 10,
        'task_repeat': 5,          # random_state를 1씩 증가시키면서 모델을 반복해서 생성함

        # 모델 생성
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

    return setting, tokenized_article_series


def get_corpus_and_dictionary(tokenized_article_series, save_path: str = 'test/'):
    try:
        # load dictionary
        dictionary = corpora.Dictionary.load(save_path + 'dictionary')
        print('-- 기존 dictionary 파일을 사용합니다.')
    except FileNotFoundError:
        # if no saved dictionary, then get new // if save_path is set then save it
        dictionary = corpora.Dictionary(tokenized_article_series)
        dictionary.save(save_path + 'dictionary') if save_path is not None else None
        print('-- 새로 dictionary 파일을 생성합니다.')

    print('dictionary size : %d' % len(dictionary))

    try:
        # load corpus
        corpus = corpora.MmCorpus(save_path + 'corpus')
        print('-- 기존 corpus 파일을 사용합니다.')
    except FileNotFoundError:
        # if no saved corpus, then get new // if save_path is set then save it
        corpus = [dictionary.doc2bow(text) for text in tokenized_article_series]
        corpora.MmCorpus.serialize(save_path + 'corpus', corpus) if save_path is not None else None
        print('-- 새로 corpus 파일을 생성합니다.')

    print('corpus size : %d' % len(corpus))

    return corpus, dictionary


def save_topics_csv(lda_model, num_topics, save_result_to: str = 'test/lda_topics.csv'):
    # LDA 모델의 토픽 리스트를 csv파일로 저장

    topics = pd.Series(lda_model.print_topics(num_topics=num_topics, num_words=10))
    # topic	list
    # 0	    (0, '0.124*"키워드" + 0.084*"키워드" + 0.067*"키워드" + ... )
    # 1	    (1, '0.077*"키워드" + 0.077*"키워드" + 0.056*"키워드" + ... )
    # 2	    (2, '0.042*"키워드" + 0.037*"키워드" + 0.034*"키워드" + ... )
    topics.to_csv(save_result_to, mode='w', encoding='utf-8', header=['list'], index_label='topic')


def get_topic_distribution_for_each_doc(lda_model, corpus):
    # 각 문서에 대한 토픽 분포를 출력하는 코드(즉, 특정 문서에 대한 모델의 토픽 예측 코드)
    # https://dianakang.tistory.com/50
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(lda_model[corpus]):
        doc = topic_list[0] if lda_model.per_word_topics else topic_list
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), (2번 토픽, 48.5%)
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48% > 25% > 21% > 5% 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc):  # 몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_list]),
                                                 ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break

    topic_table.columns = ['dominant topic number', 'dominant topic weight', 'topic number and weight']
    return topic_table


def save_lda_html(lda_model, corpus, dictionary, save_result_to: str = 'test/lda_output.html'):
    # "LDA 시각화 결과를 html파일로 저장

    output = gensim_vis.prepare(lda_model, corpus, dictionary, doc_topic_dist=None, sort_topics=False)
    # sort_topics=False의 경우 LDA 모델의 토픽 순서와 같음
    # sort_topics=True의 경우 topic portion이 높은 순으로 정렬됨
    pyLDAvis.save_html(output, save_result_to)


def lda_modeling(setting=None, tokenized_article_series=None):
    tqdm.pandas()

    # setting
    if setting or tokenized_article_series is None:
        setting, tokenized_article_series = _setting()

    corpus, dictionary = get_corpus_and_dictionary(tokenized_article_series, setting['result_dir'])
    iterations, random_state = setting['iterations'], setting['random_state']

    # LDA modeling + save model
    for i in tqdm([setting['num_topics'] for _ in range(setting['task_repeat'])]):
        try:
            lda_model = LdaModel.load(setting['result_model_dir'] + f'lda_k{i}_rd_{random_state}')
            print(f'\n해당 모델({i}개 토픽, random_state: {random_state})은 기존에 생성한 것을 사용합니다.')
        except FileNotFoundError:
            lda_model = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary,
                                 passes=20, iterations=iterations, random_state=random_state)
            lda_model.save(setting['result_model_dir'] + f'lda_k{i}_rd_{random_state}')

        save_topics_csv(lda_model, i, setting['result_dir'] + f'lda_k_{i}_rd_{random_state}.csv')
        save_lda_html(lda_model, corpus, dictionary, setting['result_dir'] + f'lda_k_{i}_rd_{random_state}.html')

        topic_table = get_topic_distribution_for_each_doc(lda_model,corpus)
        topic_table.to_csv(setting['result_dir'] + f'lda_k_{i}_rd_{random_state}_topic_table.csv',
                           mode='w', encoding='utf-8')

        # 동일한 토픽갯수를 반복하므로 random_state를 수정
        random_state += 1


def main():
    with recorder.WithTimeRecorder('LDA 모델링'):
        lda_modeling()


if __name__ == '__main__':
    main()
