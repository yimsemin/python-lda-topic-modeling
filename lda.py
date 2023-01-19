from tqdm import tqdm

import pyLDAvis
import pyLDAvis.gensim_models as gensim_vis
from gensim import corpora
from gensim.models import LdaModel

import util.openxlsx as openxlsx
import util.recorder as recorder


def _setting():
    xlsx_name: str = "input/test.xlsx"
    column_name: str = "article"
    sheet_name = "preprocessed_result"

    save_result_directory = "result/"

    num_topics: int = 7
    task_repeat: int = 15             # set_task 2일 경우, 몇 회 반복할 것인지?

    # 여러 개의 토픽 갯수를 구할 경우
    topic_number_start: int = 2
    topic_number_end: int = 60
    topic_number_interval: int = 1

    iterations: int = 20
    random_state: int = 4190

    # 수행할 작업 지정
    set_task: int = 2
    # 1 = 지정된 토픽 갯수의 LDA model, 토픽들의 txt, 시각화된 html를 1회 저장
    # 2 = 지정된 토픽 갯수의 LDA model, 토픽들의 txt, 시각화된 html를 n회 저장
    # 3 = 지정된 토픽 갯수 범위의 LDA model, 토픽들의 txt, 시각화된 html를 각각 저장

    tokenized_article_series = openxlsx.load_series_from_xlsx(xlsx_name,
                                                              column_name,
                                                              sheet_name=sheet_name,
                                                              is_list_in_list=True)

    num_topics_range = []
    if set_task == 1:
        num_topics_range = [num_topics]
    elif set_task == 2:
        num_topics_range = [num_topics for _ in range(task_repeat)]
    elif set_task == 3:
        num_topics_range = list(range(topic_number_start, topic_number_end + 1, topic_number_interval))

    return tokenized_article_series, iterations, random_state,\
        save_result_directory, num_topics_range, set_task


def get_corpus_and_dictionary(tokenized_article_series):
    dictionary = corpora.Dictionary(tokenized_article_series)
    print('dictionary size : %d' % len(dictionary))

    corpus = [dictionary.doc2bow(text) for text in tokenized_article_series]
    print('corpus size : %d' % len(corpus))

    return corpus, dictionary


def get_lda_model(corpus, dictionary, how_many_topic: int, iterations: int = 100, random_state: int = 4190):
    lda_model = LdaModel(corpus,
                         num_topics=how_many_topic,
                         id2word=dictionary,
                         passes=20,
                         iterations=iterations,
                         random_state=random_state)

    return lda_model


def save_topics_txt(lda_model, num_topics, save_result_to: str = 'result/lda_topics.txt'):
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
    lda_model = LdaModel.load(load_from_here)
    return lda_model


if __name__ == '__main__':
    tqdm.pandas()

    TOKENIZED_ARTICLE_SERIES, ITERATIONS, RANDOM_STATE, \
        SAVE_RESULT_DIRECTORY, NUM_TOPICS_RANGE, SET_TASK = _setting()

    CORPUS, DICTIONARY = get_corpus_and_dictionary(TOKENIZED_ARTICLE_SERIES)

    with recorder.WithTimeRecorder('LDA 분석'):
        initial_k = NUM_TOPICS_RANGE[0]

        for i in tqdm(NUM_TOPICS_RANGE):
            LDA_MODEL = get_lda_model(CORPUS, DICTIONARY, i, ITERATIONS, RANDOM_STATE)
            save_topics_txt(LDA_MODEL, i, SAVE_RESULT_DIRECTORY + 'lda_k_%d_rd_%d.txt' % (i, RANDOM_STATE))
            save_lda_html(LDA_MODEL, CORPUS, DICTIONARY,
                          SAVE_RESULT_DIRECTORY + 'lda_html_k_%d_rd_%d.html' % (i, RANDOM_STATE))
            save_lda_model(LDA_MODEL, SAVE_RESULT_DIRECTORY + 'lda_model_k_%d_rd_%d' % (i, RANDOM_STATE))
            if i == initial_k:
                RANDOM_STATE += 1

    print('끝')
