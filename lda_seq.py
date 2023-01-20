import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import patsy
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


def get_theta_for_each_article_each_topic(lda_seq_model, time_series):
    # 각 문서별로 토픽 분포도 값 구하기

    theta_value = lda_seq_model.doc_topics(0).tolist()
    theta_values = pd.DataFrame(columns=['id'] + ['year'] + ['topic%d' % i for i in range(1, len(theta_value) + 1)])

    # header만 있는 csv 파일 만들기
    theta_values.to_csv('test/test.csv', index=False, mode='w', encoding='utf-8')

    # 문서별 토픽 분포도 값 채워넣기
    article_counter = 0
    while True:
        try:
            print('%d번째 토픽~!' % article_counter)
            theta_value = lda_seq_model.doc_topics(article_counter).tolist()
            theta_values.loc[0] = [article_counter, time_series[article_counter], *theta_value]
            # To-Do
            # id 와 year가 float로 뒤에 .0이 붙음
            # 정수로 바꾸면 깔끔할텐데 방법을 알아보자

            theta_values.to_csv('test/test.csv', index=False, mode='a', encoding='utf-8', header=False)
            article_counter += 1
        except IndexError:
            break


def results_summary_to_dataframe(results):
    # https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe

    results_df = pd.DataFrame({"F_value": results.fvalue,
                               "F_p_value": results.f_pvalue,
                               "r_squared": results.rsquared,
                               "co_eff": results.params,
                               "std_err": results.bse,
                               "t_value": results.tvalues,
                               "p_value": results.pvalues,
                               "conf_lower": results.conf_int()[0],
                               "conf_higher": results.conf_int()[1]
                               }
                              )

    # 상수(Intercept) 제외 후 결과 출력
    return results_df.drop(['Intercept'])


def compute_each_topic_linear_regression(num_topics):

    csv_df = pd.read_csv('test/test.csv')

    results_df = pd.DataFrame(columns=["F_value", "F_p_value", "r_squared",
                                       "co_eff", "std_err", "t_value", "p_value",
                                       "conf_lower", "conf_higher"])
    results_df.to_csv('test/test_reg_result.csv', mode='w', encoding='utf-8')
    topic = 1
    while True:
        try:
            reg_result = sm.OLS.from_formula("year ~ topic%d" % topic, csv_df).fit()
            results_df = results_summary_to_dataframe(reg_result)
            results_df.to_csv('test/test_reg_result.csv', mode='a', encoding='utf-8', header=False)

            topic += 1
        except patsy.PatsyError:
            break

    # 전체 목록을 보고 싶은 경우
    # print(result.summary())


def save_seq_topics_evolution_txt(lda_seq_model, num_topics):
    with recorder.WithTxtRecorder as recorder.sys.stdout:
        for i in num_topics:
            print('[[---- %d 번 토픽 ----]]'%i)
            lda_seq_model.print_topic_times(topic=i)
            print('\n\n')


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


def main():
    tokenized_article_series, num_topics, iterations, random_state, \
        save_result_directory, time_series, time_slice = _setting()

    # lda_seq_model = get_lda_seq_model(tokenized_article_series, time_slice, num_topics, random_state, iterations)

    # save_lda_seq_model(lda_seq_model, 'result_seq/test')

    lda_seq_model = load_lda_seq_model('result_seq/test')

    # get_theta_for_each_article_each_topic(lda_seq_model, time_series)

    compute_each_topic_linear_regression(num_topics)


if __name__ == '__main__':
    main()
