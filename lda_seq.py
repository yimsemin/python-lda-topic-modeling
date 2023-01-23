import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import patsy
from gensim.models import LdaSeqModel

import lda
import util.openxlsx as openxlsx
import util.recorder as recorder


def _setting():
    # input
    input_data = {
        'xlsx_name': 'input/test.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',

        # LDA_시계열 셋팅 // 데이터가 시간순으로 정렬되어있어야 함.
        'sheet_name_seq': "preprocessed",
        'column_name_seq': "time",
        'time_format': "%Y%m"
        # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
        # date 형식으로 입력되어있어야 적용됨. 기타 알수 없는 내용일 경우 텍스트 그대로 적용
    }

    # output
    output_data = {
        'result_dir': 'result_seq/',
        'result_model_dir': 'result_seq/',
        'set_task': 1
        # 1 = Hot and Cold
        # 2 = 저장된 LDA_seq 모델을 불러와서 시각화하기
    }

    # model setting
    model_data = {
        'num_topics': 12,

        # 모델 생성
        'iterations': 20,
        'random_state': 4190,
    }

    tokenized_article_series = openxlsx.load_series_from_xlsx(input_data['xlsx_name'],
                                                              input_data['column_name'],
                                                              input_data['sheet_name'],
                                                              is_list_in_list=True)
    time_series = openxlsx.load_series_from_xlsx(input_data['xlsx_name'],
                                                 input_data['column_name_seq'],
                                                 input_data['sheet_name_seq'])
    try:
        time_series = time_series.dt.strftime(input_data['time_format'])
        print('time_format을 적용합니다')
    except AttributeError:
        print('datatime format이 아니므로, time_format을 적용하지는 않습니다.')

    time_slice = time_series.tolist()
    print(f'time_slice === \n{time_slice}')

    return input_data, output_data, model_data, tokenized_article_series, time_series, time_slice


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
                                em_min_iter=10,         # default = 6  10
                                em_max_iter=30,         # default = 20  30
                                chunksize=120)          # default = 100  120

    return lda_seq_model


def get_theta_for_each_article_each_topic(lda_seq_model, time_series) -> pd.DataFrame:

    # {'아티클 번호: int': [yyyy, topic1_theta, topic2_theta, topic3_theta, ... ]}
    theta_values = {}
    article_counter = 0
    while True:
        try:
            theta_value = lda_seq_model.doc_topics(article_counter).tolist()
            theta_values[f'{article_counter + 1}'] = [time_series[article_counter], *theta_value]

            article_counter += 1
        except IndexError:
            break

    # DataFrame으로 만들기
    header = ['year', *[f'topic{i}' for i in range(1, len(theta_values['1']))]]
    result = pd.DataFrame.from_dict(theta_values, orient='index', columns=header)

    return result


def get_linear_regression_results(reg_model) -> pd.DataFrame:
    """

    Args:
        reg_model: sm.OLS.from_formula('y ~ x', data).fit()

    Returns:
        상수 제외 x -> y 의 주요 통계값들

    """
    # https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe
    # 전체 목록을 보고 싶은 경우
    # print(result.summary())

    reg_result = pd.DataFrame({"F_value": reg_model.fvalue,
                               "F_p_value": reg_model.f_pvalue,
                               "r_squared": reg_model.rsquared,
                               "co_eff": reg_model.params,
                               "std_err": reg_model.bse,
                               "t_value": reg_model.tvalues,
                               "p_value": reg_model.pvalues,
                               "conf_lower": reg_model.conf_int()[0],
                               "conf_higher": reg_model.conf_int()[1]
                               }
                              )

    # 상수(Intercept) 제외 후 결과 출력
    return reg_result.drop(['Intercept'])


def compute_each_topic_linear_regression(theta_df):

    # theta_df = pd.read_csv('test/test.csv')

    reg_results = pd.DataFrame()
    topic = 1
    while True:
        try:
            reg_model = sm.OLS.from_formula(f'year ~ topic{topic}', theta_df).fit()
            reg_result = get_linear_regression_results(reg_model)
            reg_results = pd.Concat(reg_results, reg_result)

            topic += 1
        except patsy.PatsyError:
            break

    return reg_results


def save_seq_topics_evolution_txt(lda_seq_model, num_topics):
    # TODO
    with recorder.WithTxtRecorder as recorder.sys.stdout:
        for i in num_topics:
            print('[[---- %d 번 토픽 ----]]' % i)
            lda_seq_model.print_topic_times(topic=i)
            print('\n\n')


def save_topic_evolution_graph(df, save_graph_to: str = 'result/topic_evolution.png'):
    # TODO
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


def main():
    # setting
    _, output_data, model_data, tokenized_article_series, time_series, time_slice = _setting()

    iterations, random_state = model_data['iterations'], model_data['random_state']

    # LDA seq modeling
    try:
        lda_seq_model = LdaSeqModel.load(output_data['result_model_dir']
                                         + f'lda_k_{model_data["num_topics"]}_rd_{random_state}')
    except FileNotFoundError:
        print('새롭게 모델을 만듭니다.')
        print('이 작업은 시간이 꽤 오래걸립니다. 정말로...')
        lda_seq_model = get_lda_seq_model(tokenized_article_series, time_slice,
                                          model_data['num_topics'], random_state, iterations)
        print('모델을 저장합니다.')
        lda_seq_model.save(output_data['result_model_dir'] + f'lda_k_{model_data["num_topics"]}_rd_{random_state}')

    # Hot and Cold
    theta_df = get_theta_for_each_article_each_topic(lda_seq_model, time_series)
    theta_df.to_csv(output_data['result_dir'] + 'lda_seq_theta.csv', index=True, index_label='id', mode='w')

    reg_results = compute_each_topic_linear_regression(theta_df)
    reg_results.to_csv(output_data['result_dir'] + 'lda_seq_hot_and_cold.csv', index=True, index_label='id', mode='w')


if __name__ == '__main__':
    main()
