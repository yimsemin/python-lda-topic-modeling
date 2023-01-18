import matplotlib.pyplot as plt
from tqdm import tqdm

from gensim.models import CoherenceModel

import lda                          # 자체 제작 py
import util.openxlsx as openxlsx    # 자체 제작 py
import util.recorder as recorder    # 자체 제작 py


def _setting():
    xlsx_name = "input/6-12article.xlsx"
    column_name = 'article'
    sheet_name = "preprocessed_result"

    load_model_from: str = "result/lda_model_topic_num_"
    save_result_directory = "result/"

    min_topic_num: int = 2
    max_topic_num: int = 60
    interval_topic_num: int = 1

    # 수행할 작업 지정
    set_task: int = 1
    # 1 =

    tokenized_series = openxlsx.load_series_from_xlsx(xlsx_name, column_name, sheet_name, is_list_in_list=True)

    return tokenized_series, load_model_from, save_result_directory,\
        min_topic_num, max_topic_num, interval_topic_num, set_task


def get_perplexity(lda_model, corpus):
    # 복잡도
    # 낮을수록 좋으나 토픽 갯수가 많아지면 낮아지는 경향을 보임
    # 급격하게 낮아지는 구간을 보는 것도 방법

    return lda_model.log_perplexity(corpus)


def get_coherence(lda_model, texts, dictionary):
    # 응집도
    # 높을수록 의미론적 일관성이 높아서 좋음
    # 일정 수준 이상으로 잘 올라가지 않음, 요동치는 구간의 시작점을 보는 듯

    coherence_model_lda = CoherenceModel(model=lda_model,
                                         texts=texts,
                                         dictionary=dictionary,
                                         coherence='c_v',
                                         topn=10)
    # topn (int, optional) – 토픽을 대표하는 단어들 중, 상위 n개의 중요성을 가지는 단어를 반환
    coherence_score = coherence_model_lda.get_coherence()

    return coherence_score


def draw_plot(plot_body, range_start: int = 2, range_end: int = 15,
              x_label_name: str = 'x', y_label_name: str = 'y', save_graph_to: str = 'none'):
    x_range = range(range_start, range_end)
    plt.plot(x_range, plot_body)
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.tight_layout()

    if save_graph_to == 'none':
        plt.show()
    else:
        plt.savefig(save_graph_to)

    plt.clf()


def get_perplexity_and_coherence_value_list(tokenized_series, corpus, dictionary,
                                            min_topic_num: int = 2, max_topic_num: int = 15,
                                            interval_topic_num: int = 1,
                                            save_result_directory: str = 'result/',
                                            load_model_from: str = 'result/lda_model_'):
    perplexity_values = []
    coherence_values = []

    for i in tqdm(range(min_topic_num, max_topic_num + 1, interval_topic_num)):
        try:
            lda_model = lda.load_lda_model(load_model_from + '%d' % i)
        except FileNotFoundError:
            print('>> 토픽 갯수 %d개의 lda_model을 새로 생성합니다.' % i)
            lda_model = lda.get_lda_model(corpus, dictionary, how_many_topic=i)
            lda.save_lda_model(lda_model, save_result_directory + 'lda_model_topic_number_%d' % i)
            lda.save_lda_html(lda_model, corpus, dictionary,
                              save_result_directory + 'lda_html_topic_number_%d.html' % i)

        perplexity_values.append(get_perplexity(lda_model, corpus))
        coherence_values.append(get_coherence(lda_model, tokenized_series, dictionary))

    with recorder.WithTxtRecorder(save_result_directory + 'lda_perplexity_value.txt') as recorder.sys.stdout:
        print(*perplexity_values, sep='\n')
    with recorder.WithTxtRecorder(save_result_directory + 'lda_coherence_values.txt') as recorder.sys.stdout:
        print(*coherence_values, sep='\n')

    return perplexity_values, coherence_values


def save_perplexity_and_coherence_graph(perplexity_values, coherence_values,
                                        min_topic_num: int = 2, max_topic_num: int = 15,
                                        save_result_directory: str = 'result/',
                                        load_value_directory: str = None):

    if perplexity_values is None:
        with open(load_value_directory + 'lda_perplexity_value.txt') as f:
            perplexity_values = f.readlines()

    if coherence_values is None:
        with open(load_value_directory + 'lda_coherence_value.txt') as f:
            coherence_values = f.readlines()

    draw_plot(perplexity_values, min_topic_num, max_topic_num + 1, 'Number of topics', 'Perplexity score',
              save_result_directory + 'lda_perplexity_value.png')
    draw_plot(coherence_values, min_topic_num, max_topic_num + 1, 'Number of topics', 'Coherence value',
              save_result_directory + 'lda_coherence_value.png')


if __name__ == '__main__':
    TOKENIZED_SERIES, LOAD_MODEL_FROM, SAVE_RESULT_DIRECTORY,\
        MIN_TOPIC_NUM, MAX_TOPIC_NUM, INTERVAL_TOPIC_NUM, SET_TASK = _setting()

    CORPUS, DICTIONARY = lda.get_corpus_and_dictionary(TOKENIZED_SERIES)

    with recorder.WithTimeRecorder('최적의 토픽을 구하면서 모델, 그래프, 토픽들 전부 저장합니다.'):
        PERPLEXITY_VALUES, COHERENCE_VALUES\
            = get_perplexity_and_coherence_value_list(TOKENIZED_SERIES, CORPUS, DICTIONARY,
                                                      MIN_TOPIC_NUM, MAX_TOPIC_NUM, INTERVAL_TOPIC_NUM,
                                                      SAVE_RESULT_DIRECTORY, LOAD_MODEL_FROM)
        save_perplexity_and_coherence_graph(PERPLEXITY_VALUES, COHERENCE_VALUES,
                                            MIN_TOPIC_NUM, MAX_TOPIC_NUM,
                                            SAVE_RESULT_DIRECTORY, SAVE_RESULT_DIRECTORY)
