""" 시간과 토픽분포도의 회귀분석을 통해 토픽의 논의 추세를 파악함
"""
import datetime
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from gensim.models import LdaModel

import lda
import util.recorder as recorder
import util.token_parser as token_parser


def _setting():
    setting = {
        'lda_model': 'test/output/model/lda_k_10_rd_4190',
        'xlsx_name': 'test/input/data.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',
        'sheet_name_seq': 0,
        'column_name_seq': "date",
        'result_dir': 'test/output/',
        'reuse_saved_corpus': True
    }

    lda_model = _load_lda_model(setting)
    corpus = _load_corpus(setting)
    time_series = _read_time_series(setting)

    return setting, lda_model, corpus, time_series


def _load_lda_model(setting):
    return LdaModel.load(setting['lda_model'])


def _read_tokenized_article_series(setting):
    excel_data = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name'])[setting['column_name']]
    return token_parser.parse_tokenized_series(excel_data)


def _load_corpus(setting):
    # get corpus
    tokenized_article_series = _read_tokenized_article_series(setting)
    corpus, _ = lda.get_corpus_and_dictionary(tokenized_article_series, setting['result_dir'],
                                              setting.get('reuse_saved_corpus', True))
    return corpus


def _read_time_series(setting):
    time_series = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name_seq'])[setting['column_name_seq']]
    return time_series


def _get_date_series(time_series: pd.Series, log: bool = True) -> pd.Series:
    """ 입력된 시계열 값을 날짜로 복원한다. 복원 실패값은 NaT로 남긴다. """
    type_count = {}
    invalid_values = []

    def parse_with_format(text, time_format, type_name):
        try:
            return pd.to_datetime(text, format=time_format, errors='raise').normalize(), type_name
        except (TypeError, ValueError):
            return pd.NaT, None

    def parse_serial_date(number):
        serial_date = int(float(number))
        # 엑셀 serial date는 2173년 이후 6자리가 되므로, 그 이후 날짜를 쓰려면 이 조건을 확장해야 한다.
        return pd.to_datetime(serial_date, unit='D', origin='1899-12-30').normalize()

    def parse_one(value):
        if value is None or pd.isna(value):
            return pd.NaT, None

        if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
            return pd.Timestamp(value).normalize(), '엑셀 날짜 서식'

        text = str(value).strip()
        if not text:
            return pd.NaT, None

        if text.replace('.', '', 1).isdigit():
            number_text = text
            if '.' in text:
                number = float(text)
                number_without_time = int(number)
                number_text = str(number_without_time)
                if number != number_without_time and len(number_text) == 5:
                    return parse_serial_date(number), '엑셀 serial date(소수점 버림)'
                if number != number_without_time:
                    return pd.NaT, None

            if number_text.isdigit():
                if len(number_text) == 5:
                    return parse_serial_date(number_text), '엑셀 serial date'
                if len(number_text) == 8:
                    return parse_with_format(number_text, '%Y%m%d', 'YYYYMMDD')
                if len(number_text) == 6:
                    parsed_date, type_name = parse_with_format(number_text, '%y%m%d', 'YYMMDD')
                    if pd.notna(parsed_date):
                        return parsed_date, type_name
                    return parse_with_format(number_text, '%Y%m', 'YYYYMM')
                return pd.NaT, None

        for time_format, type_name in [('%Y-%m-%d', 'YYYY-MM-DD'), ('%Y/%m/%d', 'YYYY/MM/DD')]:
            parsed_date, parsed_type_name = parse_with_format(text, time_format, type_name)
            if pd.notna(parsed_date):
                return parsed_date, parsed_type_name

        return pd.NaT, None

    parsed_dates = {}
    for i, value in pd.Series(time_series).items():
        parsed_date, type_name = parse_one(value)
        parsed_dates[i] = parsed_date

        if type_name is None:
            invalid_values.append((i, value))
        else:
            type_count[type_name] = type_count.get(type_name, 0) + 1

    if log:
        if type_count:
            print('-- 날짜 해석 결과: ' + ', '.join([f'{key} {value}건' for key, value in type_count.items()]))
        else:
            print('-- 날짜 해석 결과: 해석 성공 없음')
        if invalid_values:
            print(f'-- 날짜 해석 실패: {len(invalid_values)}건')
            for i, value in invalid_values[:5]:
                print(f'---- 행 {i}: {value}')

    return pd.Series(parsed_dates, name=time_series.name)


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
        topic_num_and_theta_values = sorted(topic_num_and_theta_values, key=lambda x: x[0])
        # [(0, 0.0002553786), (1, 0.006252744), (2, 0.0002553786), (3, 0.0002553786), ... ]

        # 가장 theta 값이 높은 토픽을 도출
        dominant_topic_num = max(topic_num_and_theta_values, key=lambda x: x[1])[0]
        dominant_topics[i] = f'topic{dominant_topic_num}'

        # theta 값만 토픽 순으로 뽑아내기
        theta_values[i] = [theta for _, theta in topic_num_and_theta_values]
        # { 0 : [0.0002553786, 0.006252744, 0.0002553786, 0.0002553786, 0.0002553786, ... ],
        #   1 : [] [] ... ,
        #   2 : [] [] ... }

    # 문서별 가장 비중이 높은 토픽을 pd.Series로 저장
    dominant_topics_series = pd.Series(dominant_topics, name='dominant_topic')
    # 0    topic12
    # 1     topic4
    # 2    topic17
    # 3     topic4
    # ...
    # Name: dominant_topic, dtype: object

    # 문서별 토픽별 theta 값을 DataFrame으로 저장
    header = [f'topic{i}' for i in range(len(theta_values[0]))]
    theta_values_df = pd.DataFrame.from_dict(theta_values, orient='index', columns=header)
    #          topic0    topic1    topic2  ...   topic17   topic18   topic19
    # 0      0.000255  0.006253  0.000255  ...  0.000255  0.000255  0.000255
    # 1      0.000981  0.000981  0.000981  ...  0.000981  0.000981  0.000981
    # 2      0.000336  0.000336  0.000336  ...  0.532007  0.000336  0.000336
    # ...         ...       ...       ...  ...       ...       ...       ...

    return theta_values_df, dominant_topics_series


def get_example_for_each_topic(f_path='test/output/time_and_theta.csv',
                               save_result_to='test/output/example_article.txt',
                               topic_start_num=0, topic_last_num=20):
    # 각 토픽별 대표 문서 추출

    df = pd.read_csv(f_path)

    with recorder.WithTxtRecorder(save_result_to) as recorder.sys.stdout:
        for i in range(topic_start_num, topic_last_num):
            print(f'===== topic{i} =====')
            my_series = df[f'topic{i}']
            bbb = my_series.sort_values(ascending=False)
            bbb = tuple(zip(bbb, bbb.index))[0:10]

            for value, number in bbb:
                print(f'{number + 2}번째 기사, value = {value}')
            print('=====\n\n')


def get_linear_regression_results(reg_model) -> pd.DataFrame:
    """ result.summary() 에서 회귀분석이 통계적으로 유의한지 확인하는데 필요한 값들만 추출

    Args:
        reg_model: sm.OLS(y, x).fit()

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
    intercept_names = [name for name in ['Intercept', 'const'] if name in reg_result.index]
    return reg_result.drop(intercept_names)


def check_hot_and_cold(time_and_theta_csv: str, column_name_seq: str = 'time'):
    # 토픽별 회귀분석
    df = pd.read_csv(time_and_theta_csv, dtype={column_name_seq: str})
    date_series = _get_date_series(df[column_name_seq])
    valid_date_count = date_series.notna().sum()
    print(f'-- 회귀분석 사용 문서: {valid_date_count}건 / 제외: {len(df) - valid_date_count}건')
    if valid_date_count < 2:
        raise ValueError(f'{column_name_seq} 열에 분석 가능한 날짜가 2개 이상 필요합니다.')

    df = df.loc[date_series.notna()].copy()
    date_series = date_series.loc[date_series.notna()]
    if date_series.nunique() < 2:
        raise ValueError(f'{column_name_seq} 열에 서로 다른 날짜가 2개 이상 필요합니다.')

    # 선형회귀에는 연속형 숫자가 필요하므로 첫 날짜로부터 며칠 지났는지로 변환한다.
    df[column_name_seq] = (date_series - date_series.min()).dt.days

    reg_results = pd.DataFrame()
    topic = 0
    while f'topic{topic}' in df.columns:
        topic_col = f'topic{topic}'
        x = sm.add_constant(df[[column_name_seq]], has_constant='add')
        reg_model = sm.OLS(df[topic_col], x).fit()
        reg_result = get_linear_regression_results(reg_model)

        # 표가 보기 좋도록 토픽명 추가
        reg_result.insert(0, 'y', [topic_col] * len(reg_result))
        reg_results = pd.concat([reg_results, reg_result], axis=0)

        topic += 1

    if reg_results.empty:
        return reg_results

    # Hot/Cold 표기
    reg_results['Hot.Cold'] = reg_results['co_eff'].apply(lambda x: 'Hot' if x > 0 else 'Cold')
    reg_results.loc[reg_results['p_value'] > 0.05, 'Hot.Cold'] = '-'                                # p 값 설정

    return reg_results


def save_hot_and_cold_plot(time_and_theta_df: pd.DataFrame, regression_results: pd.DataFrame,
                           column_name_seq: str, save_graph_to: str):
    hot_and_cold_topics = regression_results[regression_results['Hot.Cold'].isin(['Hot', 'Cold'])]
    topic_cols = hot_and_cold_topics['y'].drop_duplicates().tolist()

    plt.figure(figsize=(10, 6))
    if topic_cols:
        date_series = _get_date_series(time_and_theta_df[column_name_seq], log=False)
        plot_df = time_and_theta_df.loc[date_series.notna(), topic_cols].copy()
        plot_df[column_name_seq] = date_series.loc[date_series.notna()]
        plot_df = plot_df.groupby(column_name_seq)[topic_cols].mean().sort_index()

        for _, row in hot_and_cold_topics.drop_duplicates('y').iterrows():
            topic_col = row['y']
            line_style = '-' if row['Hot.Cold'] == 'Hot' else '--'
            plt.plot(plot_df.index, plot_df[topic_col], marker='o', linestyle=line_style,
                     label=f'{topic_col} ({row["Hot.Cold"]})')

        plt.xlabel('Date')
        plt.ylabel('Theta')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No significant Hot/Cold topics', ha='center', va='center')
        plt.axis('off')

    plt.title('Hot/Cold Topic Trends')
    plt.tight_layout()
    recorder.ensure_parent_dir(save_graph_to)
    plt.savefig(save_graph_to)
    plt.clf()
    print(f'-- Hot/Cold 토픽 추세 그래프를 저장합니다: {save_graph_to}')


def lda_hot_and_cold(setting: dict = None,
                     lda_model=None,
                     corpus=None,
                     time_series=None):
    # setting
    if setting is None:
        default_setting, default_lda_model, default_corpus, default_time_series = _setting()
        setting = default_setting
        print('-- _setting() 기본 설정값을 사용합니다.')
        if lda_model is None:
            print('-- _setting() 기본 LDA 모델을 사용합니다.')
            lda_model = default_lda_model
        if corpus is None:
            print('-- _setting() 기본 corpus를 사용합니다.')
            corpus = default_corpus
        if time_series is None:
            print('-- _setting() 기본 시계열 데이터를 사용합니다.')
            time_series = default_time_series
    else:
        if lda_model is None:
            print('-- 전달된 setting의 lda_model에서 LDA 모델을 읽습니다.')
            lda_model = _load_lda_model(setting)
        if corpus is None:
            print('-- 전달된 setting의 입력 파일에서 corpus를 생성/불러옵니다.')
            corpus = _load_corpus(setting)
        if time_series is None:
            print('-- 전달된 setting의 입력 파일에서 시계열 데이터를 읽습니다.')
            time_series = _read_time_series(setting)

    # 데이터 셋팅 - 선형회귀 및 비중
    theta_values_df, dominant_topics_series = get_theta_for_each_article_each_topic(lda_model, corpus)
    time_series = pd.Series(time_series).reset_index(drop=True)
    time_series.name = setting['column_name_seq']
    if len(time_series) != len(theta_values_df):
        raise ValueError(f'문서 수({len(theta_values_df)})와 시계열 값 수({len(time_series)})가 다릅니다.')

    time_and_theta_df = pd.concat([time_series, theta_values_df, dominant_topics_series], axis=1)
    #        date    topic0    topic1  ...   topic18   topic19  dominant_topic
    # 0         6  0.000255  0.006253  ...  0.000255  0.000255         topic12
    # 1         6  0.000981  0.000981  ...  0.000981  0.000981          topic4
    # 2         6  0.000336  0.000336  ...  0.000336  0.000336         topic17
    # ...     ...       ...       ...  ...       ...       ...             ...

    # 분석한 데이터 저장
    time_and_theta_csv_path = os.path.join(setting['result_dir'], 'time_and_theta.csv')
    recorder.ensure_parent_dir(time_and_theta_csv_path)
    time_and_theta_df.to_csv(time_and_theta_csv_path, index=True, index_label='id', mode='w', encoding='utf-8')

    # 선형 회귀분석
    regression_results = check_hot_and_cold(time_and_theta_csv_path, setting['column_name_seq'])
    hot_and_cold_csv_path = os.path.join(setting['result_dir'], 'hot_and_cold.csv')
    recorder.ensure_parent_dir(hot_and_cold_csv_path)
    regression_results.to_csv(hot_and_cold_csv_path, index=True, index_label='id', mode='w', encoding='utf-8')

    hot_and_cold_png_path = os.path.join(setting['result_dir'], 'hot_and_cold.png')
    save_hot_and_cold_plot(time_and_theta_df, regression_results, setting['column_name_seq'], hot_and_cold_png_path)


def main():
    with recorder.WithTimeRecorder('lda_hot_and_cold'):
        lda_hot_and_cold()


if __name__ == '__main__':
    main()
