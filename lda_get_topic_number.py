import matplotlib.pyplot as plt
from tqdm import tqdm

from gensim.models import CoherenceModel

import lda                          # 자체 제작 py
import util.openxlsx as openxlsx    # 자체 제작 py
import util.recorder as recorder    # 자체 제작 py


def _setting():
    # input
    input_data = {
        'xlsx_name': 'input/test.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article'
    }

    # output
    output_data = {
        'result_dir': 'result/',
        'result_model_dir': 'result/'
    }

    # model setting
    model_data = {
        # 범위
        'topic_number_start': 2,
        'topic_number_end': 60,
        'topic_number_interval': 1,

        # 모델 생성
        'iterations': 20,
        'random_state': 4190
    }

    tokenized_article_series = openxlsx.load_series_from_xlsx(input_data['xlsx_name'],
                                                              input_data['column_name'],
                                                              input_data['sheet_name'],
                                                              is_list_in_list=True)

    model_data['topic_number_list'] = list(range(model_data['topic_number_start'],
                                                 model_data['topic_number_end'] + 1,
                                                 model_data['topic_number_interval']))

    return input_data, output_data, model_data, tokenized_article_series


def get_perplexity(lda_model, corpus):
    # 복잡도
    # 낮을수록 좋으나 토픽 갯수가 많아지면 낮아지는 경향을 보임
    # 급격하게 낮아지는 구간을 보는 것도 방법

    return lda_model.log_perplexity(corpus)


def get_coherence(lda_model, tokenized_article_series, dictionary):
    # 응집도
    # 높을수록 의미론적 일관성이 높아서 좋음
    # 일정 수준 이상으로 잘 올라가지 않음, 요동치는 구간의 시작점을 보는 듯

    coherence_model_lda = CoherenceModel(model=lda_model,
                                         texts=tokenized_article_series,
                                         dictionary=dictionary,
                                         coherence='c_v',
                                         topn=10)
    # topn (int, optional) – 토픽을 대표하는 단어들 중, 상위 n개의 중요성을 가지는 단어를 반환
    coherence_score = coherence_model_lda.get_coherence()

    return coherence_score


def get_perplexity_and_coherence_value_list(tokenized_article_series, corpus, dictionary,
                                            topic_number_list, iterations: int = 100, random_state: int = 4190,
                                            model_dir: str = 'result/',
                                            result_dir: str = 'result/'):
    perplexity_values = []
    coherence_values = []

    for i in tqdm(topic_number_list):
        try:
            lda_model = lda.LdaModel.load(model_dir + f'lda_k_{i}_rd_{random_state}')
        except FileNotFoundError:
            print(f'>> 토픽 갯수 {i}개의 lda_model을 새로 생성합니다.')
            lda_model = lda.get_lda_model(corpus, dictionary, i, iterations, random_state)
            lda_model.save(model_dir + f'lda_k_{i}_rd_{random_state}')
            lda.save_lda_html(lda_model, corpus, dictionary, result_dir + f'lda_k_{i}_rd_{random_state}.html')

        perplexity_values.append(get_perplexity(lda_model, corpus))
        coherence_values.append(get_coherence(lda_model, tokenized_article_series, dictionary))

    return perplexity_values, coherence_values


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


def save_perplexity_and_coherence_graph(perplexity_values, coherence_values,
                                        topic_number_list, result_dir: str = 'result/'):

    draw_plot(perplexity_values, topic_number_list[0], topic_number_list[-1], 'Number of topics', 'Perplexity score',
              result_dir + 'lda__perplexity_value.png')
    draw_plot(coherence_values, topic_number_list[0], topic_number_list[-1], 'Number of topics', 'Coherence value',
              result_dir + 'lda__coherence_value.png')


def save_perplexity_and_coherence_xlsx():
    # TODO
    pass


def main():
    # setting
    _, output_data, model_data, tokenized_article_series = _setting()

    corpus, dictionary = lda.get_corpus_and_dictionary(tokenized_article_series)
    iterations, random_state = model_data['iterations'], model_data['random_state']

    # LDA modeling + save result
    with recorder.WithTimeRecorder('모델, 그래프, 토픽들 전부 저장합니다.'):
        perplexity_values, coherence_values \
            = get_perplexity_and_coherence_value_list(tokenized_article_series, corpus, dictionary,
                                                      model_data['topic_number_list'], iterations, random_state,
                                                      output_data['result_model_dir'],
                                                      output_data['result_dir'])

        with recorder.WithTxtRecorder(output_data['result_dir'] + 'lda__perplexity_values.txt') as recorder.sys.stdout:
            print(*perplexity_values, sep='\n')
        with recorder.WithTxtRecorder(output_data['result_dir'] + 'lda__coherence_values.txt') as recorder.sys.stdout:
            print(*coherence_values, sep='\n')

        save_perplexity_and_coherence_graph(perplexity_values, coherence_values,
                                            model_data['topic_number_list'], output_data['result_dir'])


if __name__ == '__main__':
    main()
