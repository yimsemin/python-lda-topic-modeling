import sys
import time
from datetime import timedelta as _timedelta


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
