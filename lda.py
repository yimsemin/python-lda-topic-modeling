from tqdm import tqdm

import pyLDAvis
import pyLDAvis.gensim_models as gensim_vis
from gensim import corpora
from gensim.models import LdaModel

import util.openxlsx as openxlsx
import util.recorder as recorder


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
        'result_model_dir': 'result/',
        'set_task': 2
        # 1 = 지정된 토픽 갯수의 LDA model, 토픽들의 txt, 시각화된 html를 n회 저장
        # 2 = 지정된 토픽 갯수 범위의 LDA model, 토픽들의 txt, 시각화된 html를 각각 저장
    }

    # model setting
    model_data = {
        # 토픽의 갯수가 정해진 경우
        'num_topics': 7,
        'task_repeat': 15,

        # 다양한 토픽 갯수의 모델을 구할 경우
        'topic_number_start': 2,
        'topic_number_end': 60,
        'topic_number_interval': 1,

        # 모델 생성
        'iterations': 20,
        'random_state': 4190,
    }

    tokenized_article_series = openxlsx.load_series_from_xlsx(input_data['xlsx_name'],
                                                              input_data['column_name'],
                                                              input_data['sheet_name'],
                                                              is_list_in_list=True)

    topic_number_range_dic = {
        # set_task 값에 따라 작업 범위가 달라짐
        1: [output_data['num_topics'] for _ in range(output_data['task_repeat'])],
        2: list(range(model_data['topic_number_start'],
                      model_data['topic_number_end'] + 1,
                      model_data['topic_number_interval']))
    }
    output_data['topic_number_list'] = topic_number_range_dic[output_data['set_task']]

    return input_data, output_data, model_data, tokenized_article_series


def get_corpus_and_dictionary(tokenized_article_series):
    dictionary = corpora.Dictionary(tokenized_article_series)
    print('dictionary size : %d' % len(dictionary))

    corpus = [dictionary.doc2bow(text) for text in tokenized_article_series]
    print('corpus size : %d' % len(corpus))

    return corpus, dictionary


def get_lda_model(corpus, dictionary, num_topics: int, iterations: int = 100, random_state: int = 4190):
    lda_model = LdaModel(corpus,
                         num_topics=num_topics,
                         id2word=dictionary,
                         passes=20,
                         iterations=iterations,
                         random_state=random_state)

    return lda_model


def save_topics_txt(lda_model, num_topics, save_result_to: str = 'result/lda_topics.txt'):
    # TODO csv 저장으로 바꾸기
    # LDA 모델의 토픽 리스트를 텍스트파일로 저장

    topics = lda_model.print_topics(num_topics=num_topics, num_words=10)
    with recorder.WithTxtRecorder(save_result_to) as recorder.sys.stdout:
        print('분석 일자:', recorder.print_now(), "\n")
        print(*topics, sep='\n')


def save_lda_html(lda_model, corpus, dictionary, save_result_to: str = 'result/lda_output.html'):
    # "LDA 시각화 결과를 html파일로 저장

    output = gensim_vis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(output, save_result_to)


def save_lda_model(lda_model, save_to_here: str = 'result/lda_model'):
    lda_model.save(save_to_here)


def load_lda_model(load_from_here: str = 'result/lda_model'):
    return LdaModel.load(load_from_here)


def main():
    # setting
    tqdm.pandas()
    _, output_data, model_data, tokenized_article_series = _setting()

    corpus, dictionary = get_corpus_and_dictionary(tokenized_article_series)
    iterations, random_state = model_data['iterations'], model_data['random_state']

    # LDA modeling + save model
    with recorder.WithTimeRecorder('LDA 분석'):
        initial_k = output_data['topic_number_list'][0]
        for i in tqdm(output_data['topic_number_list']):
            lda_model = get_lda_model(corpus, dictionary, i, iterations, random_state)
            save_lda_model(lda_model, output_data['result_model_dir'] + f'lda_k_{i}_rd_{random_state}')
            save_topics_txt(lda_model, i, output_data['result_dir'] + f'lda_k_{i}_rd_{random_state}.txt')
            save_lda_html(lda_model, corpus, dictionary, output_data['result_dir'] + f'lda_k_{i}_rd_{random_state}.html')

            # 동일한 토픽갯수를 반복 작업할 경우 random_state를 수정
            if i == initial_k:
                random_state += 1

    print('끝')


if __name__ == '__main__':
    main()
