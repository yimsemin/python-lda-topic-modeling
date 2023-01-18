import sys
import time
from datetime import timedelta as _timedelta


class TxtRecorder:
    def __init__(self):
        if sys.stdout != sys.__stdout__:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        else:
            pass

        # self.original_sys_stdout = sys.__stdout__     # 기존 값을 저장해두는 방법
        self.txt_name = sys.stdout.name
        self.isRecording = False

    def txt_record_end(self):
        if self.isRecording:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            print(self.txt_name, "에 출력결과를 기록 끝!")
            self.isRecording = False
        else:
            print("기록 중이 아닙니다.")

    def txt_record_start(self, filename='output.txt', how='w', encoding='utf-8'):
        if self.isRecording:
            self.txt_record_end()
        else:
            pass

        print(filename, "에 출력결과를 기록 시작!")
        sys.stdout = open(filename, how, encoding=encoding)
        # sys.stdout.reconfigure(encoding='utf-8')  # python 3.7 이상
        self.txt_name = sys.stdout.name
        self.isRecording = True


class WithTxtRecorder(object):
    """ with 구문 속 작업의 출력 결과 기록
    Example:
        with WithTxtRecorder('output.txt', 'w', 'utf-8') as recorder.sys.stdout:
            print(result)
    """
    def __init__(self, filename: str = 'output.txt', how: str = 'w', encoding: str = 'utf-8'):
        self.file_name = filename
        self.how = how
        self.encoding = encoding

    def __enter__(self):
        self.f = open(self.file_name, self.how, encoding=self.encoding)
        return self.f

    def __exit__(self, *args):
        self.f.close()
        sys.stdout = sys.__stdout__


class TimeRecorder:
    def __init__(self):
        self.record_paper = {}
        self.the_time_when_this_start = time.time()

    def time_record_start(self, entry: str = 'task1'):
        print("[기록 시작]: ", entry)
        self.record_paper[entry] = time.time()

    def time_record_end(self, entry='task1'):
        if entry in self.record_paper:
            self.record_paper[entry + '_end'] = time.time()
            print("[기록 끝]: ", entry, " 소요시간:",
                  _timedelta(seconds=self.record_paper[entry+'_end'] - self.record_paper[entry]))
        else:
            print("[기록 끝]: ", entry, " 소요시간:",
                  _timedelta(seconds=self.record_paper[entry+'_end'] - self.the_time_when_this_start))

    def time_record_reset_all(self):
        self.record_paper.clear()


class WithTimeRecorder:
    """ with 구문 속 작업에 대한 시간 측정
    Example:
        with WithTimeRecorder('task1'):
            # do something
    """
    def __init__(self, entry: str = 'task1'):
        self.entry = entry

    def __enter__(self):
        print("[기록 시작]: ", self.entry)
        self.start_time = time.time()
        return self.start_time

    def __exit__(self, *args):
        print("[기록 끝]: ", self.entry, " 소요시간:",
              _timedelta(seconds=time.time() - self.start_time), "\n")


def print_now():
    return time.strftime('%Y.%m.%d - %H:%M:%S')


if __name__ == '__main__':
    pass
