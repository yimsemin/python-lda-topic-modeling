""" 전처리된 텍스트에 대한 LDA 모델링을 실시하고 시각화함
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)       # pandas의 future warning 가리기 위함

import os
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensim_vis
from gensim import corpora
from gensim.models import LdaModel
from tqdm import tqdm

import util.recorder as recorder
import util.token_parser as token_parser


def _setting():
    setting = {
        'xlsx_name': 'test/input/data.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',
        'result_dir': 'test/output/',
        'result_model_dir': 'test/output/model/',
        'reuse_saved_corpus': True,
        'reuse_saved_model': True,
        'num_topics': 10,
        'task_repeat': 5,
        'iterations': 50,
        'random_state': 4190
    }

    tokenized_article_series = _read_tokenized_article_series(setting)

    return setting, tokenized_article_series


def _read_tokenized_article_series(setting):
    excel_data = pd.read_excel(setting['xlsx_name'], sheet_name=setting['sheet_name'])[setting['column_name']]
    return token_parser.parse_tokenized_series(excel_data)


def get_corpus_and_dictionary(tokenized_article_series, save_path: str = 'test/output/',
                              reuse_saved: bool = True):
    source_series = getattr(tokenized_article_series, 'attrs', {}).get('source_series')
    tokenized_article_series = pd.Series(tokenized_article_series).map(token_parser.parse_tokenized_article)
    token_parser.log_empty_documents(
        tokenized_article_series,
        'LDA 분석 입력',
        '-- 빈 문서를 포함한 상태로 LDA 분석을 계속 진행합니다.',
        source_series=source_series
    )

    if save_path is not None:
        recorder.ensure_dir(save_path)
        dictionary_path = os.path.join(save_path, 'dictionary')
        if reuse_saved:
            try:
                # load dictionary
                dictionary = corpora.Dictionary.load(dictionary_path)
                print('-- 기존 dictionary 파일을 사용합니다.')
            except FileNotFoundError:
                dictionary = corpora.Dictionary(tokenized_article_series)
                dictionary.save(dictionary_path)
                print('-- 새로 dictionary 파일을 생성합니다.')
        else:
            # if no saved dictionary, then get new // if save_path is set then save it
            dictionary = corpora.Dictionary(tokenized_article_series)
            dictionary.save(dictionary_path)
            print('-- 새로 dictionary 파일을 생성합니다.')
    else:
        dictionary = corpora.Dictionary(tokenized_article_series)
        print('-- 새로 dictionary 파일을 생성합니다.')

    print('dictionary size : %d' % len(dictionary))

    if save_path is not None:
        corpus_path = os.path.join(save_path, 'corpus')
        if reuse_saved:
            try:
                # load corpus
                corpus = corpora.MmCorpus(corpus_path)
                print('-- 기존 corpus 파일을 사용합니다.')
            except FileNotFoundError:
                corpus = [dictionary.doc2bow(text) for text in tokenized_article_series]
                corpora.MmCorpus.serialize(corpus_path, corpus)
                print('-- 새로 corpus 파일을 생성합니다.')
        else:
            # if no saved corpus, then get new // if save_path is set then save it
            corpus = [dictionary.doc2bow(text) for text in tokenized_article_series]
            corpora.MmCorpus.serialize(corpus_path, corpus)
            print('-- 새로 corpus 파일을 생성합니다.')
    else:
        corpus = [dictionary.doc2bow(text) for text in tokenized_article_series]
        print('-- 새로 corpus 파일을 생성합니다.')

    print('corpus size : %d' % len(corpus))

    return corpus, dictionary


def get_lda_model_name(num_topics, random_state):
    return f'lda_k_{num_topics}_rd_{random_state}'


def get_legacy_lda_model_names(num_topics, random_state):
    return [f'lda_k{num_topics}_rd_{random_state}',
            f'lda_k{num_topics}_rd{random_state}']


def load_lda_model(model_dir, num_topics, random_state):
    model_names = [get_lda_model_name(num_topics, random_state)]
    model_names += get_legacy_lda_model_names(num_topics, random_state)

    for model_name in model_names:
        try:
            return LdaModel.load(os.path.join(model_dir, model_name)), model_name
        except FileNotFoundError:
            pass

    raise FileNotFoundError(os.path.join(model_dir, get_lda_model_name(num_topics, random_state)))


def save_topics_csv(lda_model, num_topics, save_result_to: str = 'test/output/lda_topics.csv'):
    # LDA 모델의 토픽 리스트를 csv파일로 저장

    topics = pd.Series(lda_model.print_topics(num_topics=num_topics, num_words=10))
    # topic	list
    # 0	    (0, '0.124*"키워드" + 0.084*"키워드" + 0.067*"키워드" + ... )
    # 1	    (1, '0.077*"키워드" + 0.077*"키워드" + 0.056*"키워드" + ... )
    # 2	    (2, '0.042*"키워드" + 0.037*"키워드" + 0.034*"키워드" + ... )
    recorder.ensure_parent_dir(save_result_to)
    topics.to_csv(save_result_to, mode='w', encoding='utf-8', header=['list'], index_label='topic')


def get_topic_distribution_for_each_doc(lda_model, corpus):
    # 각 문서에 대한 토픽 분포를 출력하는 코드(즉, 특정 문서에 대한 모델의 토픽 예측 코드)
    # https://dianakang.tistory.com/50
    topic_table_rows = []

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(lda_model[corpus]):
        doc = topic_list[0] if lda_model.per_word_topics else topic_list
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), (2번 토픽, 48.5%)
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48% > 25% > 21% > 5% 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc):  # 몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table_rows.append([int(topic_num), round(prop_topic, 4), topic_list])
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break

    return pd.DataFrame(
        topic_table_rows,
        columns=['dominant topic number', 'dominant topic weight', 'topic number and weight']
    )


def save_lda_html(lda_model, corpus, dictionary, save_result_to: str = 'test/output/lda_output.html',
                  note_messages: list = None):
    # "LDA 시각화 결과를 html파일로 저장

    recorder.ensure_parent_dir(save_result_to)
    tmp_path = save_result_to + '.tmp'

    def save_with_mds(mds):
        output = gensim_vis.prepare(lda_model, corpus, dictionary, doc_topic_dist=None,
                                    sort_topics=False, n_jobs=1, mds=mds)
        # sort_topics=False의 경우 LDA 모델의 토픽 순서와 같음
        # sort_topics=True의 경우 topic portion이 높은 순으로 정렬됨
        try:
            pyLDAvis.save_html(output, tmp_path)
            os.replace(tmp_path, save_result_to)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    try:
        save_with_mds('pcoa')
    except TypeError as e:
        error_message = str(e)
        if 'complex' not in error_message or 'JSON serializable' not in error_message:
            raise
        message = '-- LDA 시각화 좌표에 complex 값이 생성되어 mmds 방식으로 다시 저장합니다.'
        if note_messages is None:
            print(message)
        else:
            note_messages.append(message)
        save_with_mds('mmds')


def lda_modeling(setting=None, tokenized_article_series=None):
    tqdm.pandas()

    # setting
    if setting is None:
        setting, default_tokenized_article_series = _setting()
        print('-- _setting() 기본 설정값을 사용합니다.')
        if tokenized_article_series is None:
            print('-- _setting() 기본 입력 데이터를 사용합니다.')
            tokenized_article_series = default_tokenized_article_series
    elif tokenized_article_series is None:
        print('-- 전달된 setting의 입력 파일에서 LDA 분석 입력 데이터를 읽습니다.')
        tokenized_article_series = _read_tokenized_article_series(setting)

    reuse_saved_corpus = setting.get('reuse_saved_corpus', True)
    reuse_saved_model = setting.get('reuse_saved_model', True)
    if not reuse_saved_corpus and reuse_saved_model:
        print('-- corpus/dictionary를 새로 생성하므로 기존 LDA 모델도 재사용하지 않습니다.')
        reuse_saved_model = False

    corpus, dictionary = get_corpus_and_dictionary(tokenized_article_series, setting['result_dir'],
                                                  reuse_saved_corpus)
    iterations, random_state = setting['iterations'], setting['random_state']
    recorder.ensure_dir(setting['result_model_dir'])

    # LDA modeling + save model
    created_model_names = []
    reused_model_names = []
    renamed_model_names = []
    html_note_messages = []
    for task_index in tqdm(range(setting['task_repeat']), desc='LDA 모델 생성', unit='model'):
        i = setting['num_topics']
        model_name = get_lda_model_name(i, random_state)
        model_path = os.path.join(setting['result_model_dir'], model_name)
        if reuse_saved_model:
            try:
                lda_model, loaded_model_name = load_lda_model(setting['result_model_dir'], i, random_state)
                reused_model_names.append(model_name)
                if loaded_model_name != model_name:
                    lda_model.save(model_path)
                    renamed_model_names.append(model_name)
            except FileNotFoundError:
                lda_model = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary,
                                     passes=20, iterations=iterations, random_state=random_state)
                lda_model.save(model_path)
                created_model_names.append(model_name)
        else:
            lda_model = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary,
                                 passes=20, iterations=iterations, random_state=random_state)
            lda_model.save(model_path)
            created_model_names.append(model_name)

        save_topics_csv(lda_model, i, os.path.join(setting['result_dir'], f'{model_name}.csv'))
        save_lda_html(lda_model, corpus, dictionary, os.path.join(setting['result_dir'], f'{model_name}.html'),
                      html_note_messages)

        topic_table = get_topic_distribution_for_each_doc(lda_model, corpus)
        topic_table_path = os.path.join(setting['result_dir'], f'{model_name}_topic_table.csv')
        recorder.ensure_parent_dir(topic_table_path)
        topic_table.to_csv(topic_table_path, mode='w', encoding='utf-8')

        # 동일한 토픽갯수를 반복하므로 random_state를 수정
        random_state += 1

    print(f'-- LDA 모델 생성 완료: 새 모델 {len(created_model_names)}개, 기존 모델 {len(reused_model_names)}개')
    if renamed_model_names:
        print('-- 모델 파일명 정리: '+', '.join(renamed_model_names))
    if html_note_messages:
        print('-- HTML 저장 특이사항:')
        for message in html_note_messages:
            print(f'---- {message}')


def main():
    with recorder.WithTimeRecorder('LDA 모델링'):
        lda_modeling()


if __name__ == '__main__':
    main()
