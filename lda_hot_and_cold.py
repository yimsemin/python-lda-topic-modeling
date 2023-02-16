""" 시간과 토픽분포도의 회귀분석을 통해 토픽의 논의 추세를 파악함

x: 각 문서별 토픽 분포도(문서별 각 토픽일 확률 = theta / 문서의 theta 값의 합은 1)
y: 시간
y = ax + b 의 회귀분석을 실시함
"""
import pandas as pd
import patsy
import statsmodels.api as sm
from gensim.models import LdaModel

import lda
import util.recorder as recorder


def _setting():
    setting = {
        # input
        'lda_model': 'result/test/model/lda_k_12_rd_4190',

        'xlsx_name': 'input/test.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',

        'sheet_name_seq': "Sheet1",
        'column_name_seq': "date",
        'time_format': "%Y%m",
        # date 형식으로 입력되어 있다면 time_format의 형태로 바꿔줌. 기타 텍스트 및 숫자는 적용되지 않음
        # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior

        # output
        'result_dir': 'result/test/'
    }

    lda_model = LdaModel.load(setting['lda_model'])

    # get corpus
    excel_data = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name'])[setting['column_name']]
    tokenized_article_series = excel_data.map(lambda line: line.split(','), na_action='ignore')
    corpus, _ = lda.get_corpus_and_dictionary(tokenized_article_series, setting['result_dir'])

    time_series = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name_seq'])[setting['column_name_seq']]

    try:
        time_series = time_series.dt.strftime(setting['time_format'])
        print('time_format을 적용합니다')
    except AttributeError:
        print('datatime format이 아니므로, time_format을 적용하지는 않습니다.')

    return setting, lda_model, corpus, time_series


def get_theta_for_each_article_each_topic(lda_model, corpus) -> (pd.DataFrame, pd.Series):
    """ 각 문서별 각 토픽에 대한 theta 값을 pd.DataFrame으로 제시

    Args:
        lda_model:
        corpus:

    Returns:
        theta_values_df
        dominant_topics_series
    """
    theta_values = {}
    dominant_topics = {}
    for i in range(len(corpus)):
        topic_num_and_theta_values = lda_model.get_document_topics(corpus[i], 0.0)
        # [(0, 0.0002553786), (1, 0.006252744), (2, 0.0002553786), (3, 0.0002553786), ... ]

        # 가장 theta 값이 높은 토픽을 도출
        dominant_topics[i] = f'topic{sorted(topic_num_and_theta_values, key=lambda x: (x[1]), reverse=True)[0][0] + 1}'

        # theta 값만 토픽 순으로 뽑아내기
        theta_values[i] = [i[1] for i in topic_num_and_theta_values]
        # { 0 : [0.0002553786, 0.006252744, 0.0002553786, 0.0002553786, 0.0002553786, ... ],
        #   1 : [] [] ... ,
        #   2 : [] [] ... }

    # 문서별 가장 비중이 높은 토픽을 pd.Series로 저장
    dominant_topics_series = pd.Series(dominant_topics, name='dominant_topic')
    # 0    topic13
    # 1     topic5
    # 2    topic18
    # 3     topic5
    # ...
    # Name: dominant_topic, dtype: object

    # 문서별 토픽별 theta 값을 DataFrame으로 저장
    header = [f'topic{i}' for i in range(len(theta_values[0]))]
    theta_values_df = pd.DataFrame.from_dict(theta_values, orient='index', columns=header)
    #          topic1    topic2    topic3  ...   topic18   topic19   topic20
    # 0      0.000255  0.006253  0.000255  ...  0.000255  0.000255  0.000255
    # 1      0.000981  0.000981  0.000981  ...  0.000981  0.000981  0.000981
    # 2      0.000336  0.000336  0.000336  ...  0.532007  0.000336  0.000336
    # ...         ...       ...       ...  ...       ...       ...       ...

    return theta_values_df, dominant_topics_series


def get_linear_regression_results(reg_model) -> pd.DataFrame:
    """ result.summary() 에서 회귀분석이 통계적으로 유의한지 확인하는데 필요한 값들만 추출

    Args:
        reg_model: sm.OLS.from_formula('y ~ x', data).fit()

    Returns:
        상수 제외 x -> y 의 주요 통계값들
    """
    # https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe
    # 전체 결과를 보고 싶은 경우
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
                               })

    # 상수(Intercept) 제외 후 결과 출력
    return reg_result.drop(['Intercept'])


def compute_each_topic_linear_regression(time_and_theta_df: pd.DataFrame, column_name_seq: str = 'time'):
    # 토픽별 회귀분석
    reg_results = pd.DataFrame()
    topic = 0
    while True:
        try:
            reg_model = sm.OLS.from_formula(f'{column_name_seq} ~ topic{topic}', time_and_theta_df).fit()
            reg_result = get_linear_regression_results(reg_model)
            reg_results = pd.concat([reg_results, reg_result], axis=0)

            topic += 1
        # topic'n'이 존재하지 않는 경우
        except patsy.PatsyError:
            break

    return reg_results
함요

def check_hot_and_cold():
    # TODO
    pass


def lda_hot_and_cold(setting: dict = None,
                     lda_model=None,
                     corpus=None,
                     time_series=None):
    # setting
    if setting or lda_model or corpus or time_series is None:
        print('기본 셋팅으로 진행')
        setting, lda_model, corpus, time_series = _setting()

    # 데이터 셋팅 - 선형회귀 및 비중
    theta_values_df, dominant_topics_series = get_theta_for_each_article_each_topic(lda_model, corpus)
    time_and_theta_df = pd.concat([time_series, theta_values_df, dominant_topics_series], axis=1)
    #        date    topic0    topic1  ...   topic19   topic20  dominant_topic
    # 0         6  0.000255  0.006253  ...  0.000255  0.000255         topic13
    # 1         6  0.000981  0.000981  ...  0.000981  0.000981          topic5
    # 2         6  0.000336  0.000336  ...  0.000336  0.000336         topic18
    # ...     ...       ...       ...  ...       ...       ...             ...

    # 분석한 데이터 저장
    time_and_theta_df.to_csv(setting['result_dir'] + 'time_and_theta.csv', index=True, index_label='id', mode='w')

    # 선형 회귀분석
    # TODO 완전하지 않음 / time_and_theta.csv 을 별도로 spss 또는 jamovi로 회귀분석 하는 것을 추천함
    reg_results = compute_each_topic_linear_regression(time_and_theta_df, setting['column_name_seq'])
    reg_results.to_csv(setting['result_dir'] + 'hot_and_cold.csv', index=True, index_label='id', mode='w')
    # TODO Hot Cold 까지 적은 다음에 csv로 저장하기

    # Time별 각 토픽 비중 확인하기
    # TODO 우선은 구글스프레드시트로 그래프 그렸는데 파이썬으로 가능한지 찾아보기


def main():
    with recorder.WithTimeRecorder('lda_hot_and_cold'):
        lda_hot_and_cold()


if __name__ == '__main__':
    main()
