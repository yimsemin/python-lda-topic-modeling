""" 시간과 토픽분포도의 회귀분석을 통해 토픽의 논의 추세를 파악함
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
        'lda_model': 'test/model/lda_k10_rd_4190',          # 분석할 모델명을 기술

        'xlsx_name': 'test/test.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',

        'sheet_name_seq': "Sheet1",                         # 시계열 정보가 담긴 시트 이름
        'column_name_seq': "date",                          # 시계열 정보가 담긴 열 제목 (첫번째 행)
        'time_format': "%Y%m",
        # 엑셀에서 '날짜' 서식으로 입력했다면, 위 date 형식으로 바꿔줌 / 필요 없다면 엑셀에서 '텍스트' 서식으로 입력할 것
        # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior

        # output
        'result_dir': 'test/'
    }

    lda_model = LdaModel.load(setting['lda_model'])

    # get corpus
    excel_data = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name'])[setting['column_name']]
    tokenized_article_series = excel_data.map(lambda line: line.split(','), na_action='ignore')
    corpus, _ = lda.get_corpus_and_dictionary(tokenized_article_series, setting['result_dir'])

    time_series = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name_seq'])[setting['column_name_seq']]
    try:
        time_series = time_series.dt.strftime(setting['time_format'])
        print('-- time_format을 적용합니다.')
    except AttributeError:
        print('-- datatime format이 아니므로, 입력된 값을 그대로 사용합니다.')

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


def get_example_for_each_topic(f_path='test/time_and_theta.csv',
                               save_result_to='test/example_article.txt',
                               topic_start_num=0, topic_last_num=20):
    # 각 토픽별 대표 문서 추출
    # Todo time이랑 theta를 한 함수에 둘 필요가 없음... 구분 필요

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
    #       F_value  F_p_value  r_squared  ...   p_value  conf_lower  conf_higher
    # time   1.7177   0.191071   0.006141  ...  0.191071   -0.035879     0.007199
    # 위 index의 time은 독립변수 x를 의미함
    return reg_result.drop(['Intercept'])


def check_hot_and_cold(time_and_theta_csv: str, column_name_seq: str = 'time'):
    # 토픽별 회귀분석
    df = pd.read_csv(time_and_theta_csv)
    reg_results = pd.DataFrame()
    topic = 0
    while True:
        try:
            reg_model = sm.OLS.from_formula(f'topic{topic} ~ {column_name_seq}', df).fit()
            reg_result = get_linear_regression_results(reg_model)

            topic += 1

            # 표가 보기 좋도록 토픽명 추가
            reg_result.insert(0, 'y', [f'topic{topic}'])
            reg_results = pd.concat([reg_results, reg_result], axis=0)

        # topic'n'이 존재하지 않는 경우
        except patsy.PatsyError:
            break

    # Hot/Cold 표기
    reg_results['Hot.Cold'] = reg_results['co_eff'].apply(lambda x: 'Hot' if x > 0 else 'Cold')
    reg_results.loc[reg_results['p_value'] > 0.05, 'Hot.Cold'] = '-'                                # p 값 설정

    return reg_results


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
    regression_results = check_hot_and_cold(setting['result_dir'] + 'time_and_theta.csv', setting['column_name_seq'])
    regression_results.to_csv(setting['result_dir'] + 'hot_and_cold.csv', index=True, index_label='id', mode='w')

    # TODO Hot, Cold 나눠서 추세를 그래프로 시각화하기


def main():
    with recorder.WithTimeRecorder('lda_hot_and_cold'):
        lda_hot_and_cold()


if __name__ == '__main__':
    main()
