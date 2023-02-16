""" 전처리된 텍스트에 대한 LDA 모델링을 실시하고 시각화함

1. 말뭉치(corpus)와 딕셔너리(dictionary)의 생성, 저장
2. 지정된 토픽 갯수의 LDA 모델의 생성
3. 생성된 LDA 모델의 시각화
4. 생성된 LDA 모델의 각 토픽별 대표 문서 뽑아내기

"""
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
        'xlsx_name': 'input/test.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',

        # output -- 폴더는 미리 만들어둬야 함
        'result_dir': 'result/test/',
        'result_model_dir': 'result/test/model/',

        # model setting
        # 토픽의 갯수가 정해진 경우
        'num_topics': 12,
        'task_repeat': 5,          # 반복할 경우 random_state를 1씩 증가시키면서 작업을 수행함

        # 모델 생성
        'iterations': 20,
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


def get_corpus_and_dictionary(tokenized_article_series, save_path: str = None):
    try:
        # load dictionary
        dictionary = corpora.Dictionary.load(save_path + 'dictionary')
    except FileNotFoundError:
        # if no saved dictionary, then get new // if save_path is set then save it
        dictionary = corpora.Dictionary(tokenized_article_series)
        dictionary.save(save_path + 'dictionary') if save_path is not None else None

    print('dictionary size : %d' % len(dictionary))

    try:
        # load corpus
        corpus = corpora.MmCorpus(save_path + 'corpus')
    except FileNotFoundError:
        # if no saved corpus, then get new // if save_path is set then save it
        corpus = [dictionary.doc2bow(text) for text in tokenized_article_series]
        corpora.MmCorpus.serialize(save_path + 'corpus', corpus) if save_path is not None else None

    print('corpus size : %d' % len(corpus))

    return corpus, dictionary


def save_topics_csv(lda_model, num_topics, save_result_to: str = 'result/lda_topics.csv'):
    # LDA 모델의 토픽 리스트를 csv파일로 저장

    topics = pd.Series(lda_model.print_topics(num_topics=num_topics, num_words=10))
    # topic	list
    # 0	    (0, '0.124*"키워드" + 0.084*"키워드" + 0.067*"키워드" + ... )
    # 1	    (1, '0.077*"키워드" + 0.077*"키워드" + 0.056*"키워드" + ... )
    # 2	    (2, '0.042*"키워드" + 0.037*"키워드" + 0.034*"키워드" + ... )
    topics.to_csv(save_result_to, mode='w', encoding='utf-8', header=['list'], index_label='topic')


def get_topic_distribution_for_each_doc(lda_model, corpus):
    # TODO lda_hot_and_cold 랑 비교해서 무엇이 더 좋을지 판단 필요
    # 각 문서에 대한 토픽 분포를 출력하는 코드(즉, 특정 문서에 대한 모델의 토픽 예측 코드)
    # https://dianakang.tistory.com/50
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(lda_model[corpus]):
        doc = topic_list[0] if lda_model.per_word_topics else topic_list
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%),
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc):  # 몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return topic_table


def get_example_for_each_topic(f_path='result/time_and_theta.csv',
                               save_result_to='result/example_article.txt',
                               topic_start_num=0, topic_last_num=20):
    # TODO 작성 중 -- lda_hot_and_cold에서 코드 분리해야 함
    # 각 문서별 대표 기사 추출'

    df = pd.read_csv(f_path)

    with recorder.WithTxtRecorder(save_result_to) as recorder.sys.stdout:
        for i in range(topic_start_num, topic_last_num):
            print(f'===== topic{i} =====')
            my_series = df[f'topic{i}']
            bbb = my_series.sort_values(ascending=False)
            bbb = tuple(zip(bbb, bbb.index))[0:10]

            new_list = []
            for value, number in bbb:
                new_list.append(value, number)

            for value, number in bbb:
                print(f'{number + 2}번째 기사, value = {value}')
            print('=====\n\n')


def save_lda_html(lda_model, corpus, dictionary, save_result_to: str = 'result/lda_output.html'):
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
            lda_model = LdaModel.load(setting['result_model_dir'] + f'lda_k_{i}_rd_{random_state}')
        except FileNotFoundError:
            lda_model = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary,
                                 passes=20, iterations=iterations, random_state=random_state)
            lda_model.save(setting['result_model_dir'] + f'lda_k_{i}_rd_{random_state}')

        save_topics_csv(lda_model, i, setting['result_dir'] + f'lda_k_{i}_rd_{random_state}.csv')
        save_lda_html(lda_model, corpus, dictionary, setting['result_dir'] + f'lda_k_{i}_rd_{random_state}.html')

        # 동일한 토픽갯수를 반복하므로 random_state를 수정
        random_state += 1


def main():
    with recorder.WithTimeRecorder('LDA 모델링'):
        lda_modeling()


if __name__ == '__main__':
    main()
