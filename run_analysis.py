"""실제 input/output 경로를 사용하는 통합 실행 파일

각 세부 파일의 _setting()을 직접 수정하지 않고, 이 파일의 _setting()만 수정해서
전처리부터 토픽 수 탐색, 최종 모델 생성, 시계열 분석까지 실행한다.
"""

import os

import frequency_analysis as frequency
import lda
import lda_explore_topic_number as explore_topic
import lda_hot_and_cold as hot_and_cold
import preprocessing
import util.recorder as recorder


def _setting():
    return {
        # task setting
        # 1: 전처리 실행(preprocessing.py)
        # 2: 전처리 결과의 빈도분석 실행(frequency_analysis.py)
        # 3: topic_number_start~end 범위의 토픽 갯수 탐색 실행(lda_explore_topic_number.py)
        # 4: num_topics로 선택한 토픽수의 LDA 모델을 task_repeat만큼 생성(lda.py)
        # 5: 선택한 LDA 모델로 Hot/Cold 시계열 분석 실행(lda_hot_and_cold.py)
        "tasks": [1],
        # input
        "xlsx_name": "input/data.xlsx",  # 엑셀 파일은 한 행에 문서 하나가 들어가는 형식이어야 함
        "raw_sheet_name": 0,  # raw_sheet_name 0 입력 -> 가장 왼쪽 시트 선택
        "text_column_name": "article",
        "time_sheet_name": 0,  # time_sheet_name에 0 입력 -> 가장 왼쪽 시트 선택
        "time_column_name": "date",  # 날짜는 엑셀 날짜 서식, serial date, YYYYMMDD, YYMMDD, YYYY-MM-DD, YYYY/MM/DD 지원
        "stopwordlist_location": "input/stopwordlist.txt",  # 불용어는 단어/품사 형식, 품사 생략 시 NNG로 처리
        # output
        "preprocessed_sheet_name": "preprocessed",  # 전처리 결과는 입력 엑셀 파일 안의 이 시트에 저장됨. 시트가 이미 있으면 덮어씀
        "result_dir": "output/",
        "result_model_dir": "output/model/",
        "empty_document_csv_name": "output/preprocessing_empty_documents.csv",  # 전처리 후 빈 문서가 된 사례를 기록하는 파일
        "frequency_csv_name": "output/frequency_analysis.csv",
        "word_cloud_name": "output/word_cloud.png",
        "word_cloud_font": "font/NanumGothic.ttf",
        # preprocessing setting
        "preprocessing_min_word_count": 10,  # n회 이하 나타난 단어는 전처리 결과에서 제거
        # frequency setting
        "frequency_min_word_count": 10,  # n회 이하 나타난 단어는 빈도분석 결과에서 제외
        # topic number explore setting
        "topic_number_start": 2,  # 탐색할 최소 토픽 값
        "topic_number_end": 40,  # 탐색할 최대 토픽 값
        "topic_number_interval": 1,  # start부터 end까지 이 간격으로 토픽 갯수를 조사
        "topic_number_list": None,  # 예: [5, 7, 10, 12] / None이면 start~end 범위를 사용
        "save_explore_html": True,  # 탐색 단계에서 각 k별 HTML를 같이 저장
        "save_explore_topic_detail": True,  # 탐색 단계에서 각 k별 토픽/문서별 CSV를 같이 저장
        # final LDA model setting
        "num_topics": 10,  # 최종적으로 선정한 토픽 갯수
        "task_repeat": 5,  # 같은 토픽 갯수에서 random_state를 1씩 증가시키며 모델을 반복 생성
        # LDA training setting
        "iterations": 50,
        "random_state": 4190,
        # cache setting
        # 입력 엑셀을 바꾼 뒤에는 False로 두거나 기존 output을 삭제
        "reuse_saved_corpus": True,
        # 같은 토픽수/랜덤값 모델을 새로 만들려면 False
        "reuse_saved_model": True,
        # hot/cold setting
        # None이면 result_model_dir, num_topics, random_state로 모델 경로를 자동 생성
        "lda_model": None,
    }


def _preprocessing_setting(setting):
    return {
        "xlsx_name": setting["xlsx_name"],
        "sheet_name": setting["raw_sheet_name"],
        "column_name": setting["text_column_name"],
        "stopwordlist_location": setting["stopwordlist_location"],
        "result_sheet_name": setting["preprocessed_sheet_name"],
        "empty_document_csv_name": setting["empty_document_csv_name"],
        "min_word_count": setting["preprocessing_min_word_count"],
    }


def _frequency_setting(setting):
    return {
        "xlsx_name": setting["xlsx_name"],
        "sheet_name": setting["preprocessed_sheet_name"],
        "column_name": setting["text_column_name"],
        "result_csv_name": setting["frequency_csv_name"],
        "result_word_cloud_name": setting["word_cloud_name"],
        "word_cloud_font": setting["word_cloud_font"],
        "min_word_count": setting["frequency_min_word_count"],
    }


def _explore_setting(setting):
    task_setting = {
        "xlsx_name": setting["xlsx_name"],
        "sheet_name": setting["preprocessed_sheet_name"],
        "column_name": setting["text_column_name"],
        "result_dir": setting["result_dir"],
        "result_model_dir": setting["result_model_dir"],
        "save_explore_html": setting["save_explore_html"],
        "save_explore_topic_detail": setting["save_explore_topic_detail"],
        "reuse_saved_corpus": setting["reuse_saved_corpus"],
        "reuse_saved_model": setting["reuse_saved_model"],
        "topic_number_start": setting["topic_number_start"],
        "topic_number_end": setting["topic_number_end"],
        "topic_number_interval": setting["topic_number_interval"],
        "iterations": setting["iterations"],
        "random_state": setting["random_state"],
    }

    if setting["topic_number_list"] is not None:
        task_setting["topic_number_list"] = setting["topic_number_list"]

    return task_setting


def _lda_setting(setting):
    return {
        "xlsx_name": setting["xlsx_name"],
        "sheet_name": setting["preprocessed_sheet_name"],
        "column_name": setting["text_column_name"],
        "result_dir": setting["result_dir"],
        "result_model_dir": setting["result_model_dir"],
        "reuse_saved_corpus": setting["reuse_saved_corpus"],
        "reuse_saved_model": setting["reuse_saved_model"],
        "num_topics": setting["num_topics"],
        "task_repeat": setting["task_repeat"],
        "iterations": setting["iterations"],
        "random_state": setting["random_state"],
    }


def _get_lda_model_path(setting):
    if setting["lda_model"] is not None:
        return setting["lda_model"]

    model_name = lda.get_lda_model_name(setting["num_topics"], setting["random_state"])
    return os.path.join(setting["result_model_dir"], model_name)


def _hot_and_cold_setting(setting):
    return {
        "lda_model": _get_lda_model_path(setting),
        "xlsx_name": setting["xlsx_name"],
        "sheet_name": setting["preprocessed_sheet_name"],
        "column_name": setting["text_column_name"],
        "sheet_name_seq": setting["time_sheet_name"],
        "column_name_seq": setting["time_column_name"],
        "result_dir": setting["result_dir"],
        "reuse_saved_corpus": setting["reuse_saved_corpus"],
    }


def _validate_tasks(tasks, task_names):
    if isinstance(tasks, int):
        tasks = [tasks]

    unknown_tasks = [task for task in tasks if task not in task_names]
    if unknown_tasks:
        raise ValueError(f"알 수 없는 task 번호입니다: {unknown_tasks}")

    return tasks


def run_analysis(setting: dict = None):
    if setting is None:
        setting = _setting()

    task_names = {
        1: "전처리",
        2: "빈도분석",
        3: "토픽 갯수 탐색",
        4: "선택 토픽수 LDA 모델링",
        5: "Hot/Cold 시계열 분석",
    }
    tasks = _validate_tasks(setting["tasks"], task_names)

    task_labels = [f"{task}:{task_names[task]}" for task in tasks]
    print(f"-- 전체 작업: {len(tasks)}개 ({', '.join(task_labels)})", flush=True)

    with recorder.WithTimeRecorder(f"전체 분석 ({len(tasks)}개 작업)"):
        for task_index, task in enumerate(tasks, start=1):
            task_name = task_names[task]
            print(
                f"\n-- 작업 진행: {task_index}/{len(tasks)} 시작 - {task_name}",
                flush=True,
            )
            with recorder.WithTimeRecorder(f"{task_index}/{len(tasks)} {task_name}"):
                if task == 1:
                    preprocessing.preprocessing_noun(_preprocessing_setting(setting))
                elif task == 2:
                    frequency.frequency_analysis(_frequency_setting(setting))
                elif task == 3:
                    explore_topic.lda_explore_topic_number(_explore_setting(setting))
                elif task == 4:
                    lda.lda_modeling(_lda_setting(setting))
                elif task == 5:
                    hot_and_cold.lda_hot_and_cold(_hot_and_cold_setting(setting))
            print(
                f"-- 작업 진행: {task_index}/{len(tasks)} 완료 - {task_name}",
                flush=True,
            )


def main():
    run_analysis()


if __name__ == "__main__":
    main()
