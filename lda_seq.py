import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import LdaSeqModel

import lda
import util.openxlsx as openxlsx
import util.recorder as recorder


def _setting():
    xlsx_name = "input/test.xlsx"
    column_name = "article"
    sheet_name = "preprocessed_result"

    load_seq_model_from: str = "result_seq/lda_seq_model"
    save_result_directory = "result_seq/"

    num_topics: int = 7

    iterations: int = 20
    random_state: int = 4190

    # LDA_시계열 셋팅 // 데이터가 시간순으로 정렬되어있어야 합니다.
    column_name_seq = "time"
    sheet_name_seq = "preprocessed_result"

    # 수행할 작업
    set_task: int = 1
    # 1 = LDA_seq 모델을 생성하고 저장하기
    # 2 = 저장된 LDA_seq 모델을 불러와서 시각화하기

    tokenized_article_series = openxlsx.load_series_from_xlsx(xlsx_name,
                                                              column_name,
                                                              sheet_name=sheet_name,
                                                              is_list_in_list=True)

    time_series = openxlsx.load_series_from_xlsx(xlsx_name, column_name_seq, sheet_name_seq)
    time_slice = time_series.value_counts(sort=False).tolist()

    if set_task == 2:
        lda_seq_model = LdaSeqModel.load(load_seq_model_from)

    return tokenized_article_series, num_topics, iterations, random_state,\
        save_result_directory, time_series, time_slice


def get_lda_seq_model(tokenized_article_series, time_slice, num_topics, random_state, iterations):
    # https://radimrehurek.com/gensim/models/ldaseqmodel.html
    # https://markroxor.github.io/gensim/static/notebooks/ldaseqmodel.html

    corpus, dictionary = lda.get_corpus_and_dictionary(tokenized_article_series)

    lda_seq_model = LdaSeqModel(corpus,
                                time_slice=time_slice,
                                id2word=dictionary,
                                num_topics=num_topics,
                                passes=20,
                                random_state=random_state,
                                lda_inference_max_iter=iterations,
                                em_min_iter=10,         # default = 6
                                em_max_iter=30,         # default = 20
                                chunksize=120)          # default = 100

    return lda_seq_model


def save_seq_topics_evolution_txt(lda_seq_model, num_topics):
    with recorder.WithTxtRecorder as recorder.sys.stdout:
        for i in num_topics:
            print('[[---- %d 번 토픽 ----]]'%i)
            lda_seq_model.print_topic_times(topic=i)
            print('\n\n')


def get_article_influence(lda_seq_model, num_topics, time_slice: list):
    article_list, topic_id, period, distributions = [], [], [], []
    for topic in range(num_topics):
        for t in range(len(time_slice)):
            for article in range(time_slice[t]):
                distribution = round(lda_seq_model.influences_time[t][article][topic], 4)
                period.append(t)
                article_list.append(article)
                topic_id.append(topic)
                distributions.append(distribution)

    return pd.DataFrame(list(zip(article_list, topic_id, period, distributions)),
                        columns=['article', 'topic_id', 'period', 'distribution'])


def get_topic_distribution(lda_seq_model, num_topics, time_slice: list):
    article_list, topic_id, distributions = [], [], []
    df_dim = get_article_influence(lda_seq_model=lda_seq_model, num_topics=num_topics, time_slice=time_slice)
    for article in range(0, sum(time_slice)):
        for topic in range(0, num_topics):
            distribution = round(lda_seq_model.gamma_[article][topic], 4)
            article_list.append(article)
            topic_id.append(topic)
            distributions.append(distribution)

    return pd.DataFrame(list(zip(article_list, topic_id, distributions, df_dim.period)),
                        columns=['article', 'topic_id', 'distribution', 'period'])


def save_topic_evolution_graph(df, save_graph_to: str = 'result/topic_evolution.png'):
    fig, ax = plt.subplots(figsize=(30, 10))
    df.groupby(['period', 'topic_id'], sort=False).mean()['distribution'].unstack().plot(ax=ax,
                                                                                         grid=True,
                                                                                         linewidth=3.0,
                                                                                         sharex=True)
    plt.ylabel("Topic Distribution", fontsize=16)
    plt.xlabel("Period", fontsize=16)
    plt.title("Topic evolution")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Topics", fontsize='large', labelspacing=0.6,
               fancybox=True)

    if save_graph_to == 'none':
        plt.show()
    else:
        plt.savefig(save_graph_to)

    plt.clf()


def save_lda_seq_model(lda_seq_model, save_to_here: str = 'result/lda_seq_model'):
    lda_seq_model.save(save_to_here)


def load_lda_seq_model(load_from_here: str = 'result/lda_seq_model'):
    lda_seq_model = LdaSeqModel.load(load_from_here)

    return lda_seq_model


if __name__ == '__main__':

    TOKENIZED_ARTICLE_SERIES, NUM_TOPICS, ITERATIONS, RANDOM_STATE,\
        SAVE_RESULT_DIRECTORY, TIME_SERIES, TIME_SLICE = _setting()

    LDA_SEQ_MODEL = get_lda_seq_model(tokenized_article_series=TOKENIZED_ARTICLE_SERIES,
                                      time_slice=TIME_SLICE,
                                      num_topics=NUM_TOPICS,
                                      random_state=RANDOM_STATE,
                                      iterations=ITERATIONS)

    DF_TOPIC_DISTRIBUTION = get_topic_distribution(lda_seq_model=LDA_SEQ_MODEL,
                                                   num_topics=NUM_TOPICS,
                                                   time_slice=TIME_SLICE)

    DF_TOPIC_DISTRIBUTION.to_csv("result/topic_seq_df.csv", index=False)

    save_topic_evolution_graph(DF_TOPIC_DISTRIBUTION, save_graph_to=SAVE_RESULT_DIRECTORY + 'topic_evolution.png')

