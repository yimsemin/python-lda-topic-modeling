""" 진행시간 또는 print()을 기록하는 class

with 구문 속 작업에 대해
1) 터미널에 기록될 내용을 텍스트 파일에 기록 - with WithTxtRecorder('output.txt', 'w') as recorder.sys.stdout:
2) 코드 진행 시간을 기록 - with WithTimeRecorder('task_name'):

"""
import sys
import datetime
import time


class WithTxtRecorder(object):
    """ with 구문 속 작업에 대해, 터미널에 기록 될 내용을 텍스트 파일에 기록

    Example:
        with WithTxtRecorder('output.txt', 'w') as recorder.sys.stdout:
            print(something)

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
    """ with 구문 속 작업에 대해, 시간 측정

    Example:
        with WithTimeRecorder('task_name'):
            # do something

    """
    def __init__(self, entry: str = 'task1'):
        self.entry = entry

    def __enter__(self):
        self.start_time = time.time()
        print(f'[기록 시작]: {self.entry}\n'
              f' - 시작시간: {time.strftime("%Y.%m.%d - %H:%M:%S", time.localtime(self.start_time))}')

    def __exit__(self, *args):
        self.end_time = time.time()
        print(f'[기록 끝]: {self.entry}\n'
              f' - 종료시간: {time.strftime("%H:%M:%S")}\n'
              f' - 소요시간: {datetime.timedelta(seconds=self.end_time - self.start_time)}\n')


if __name__ == '__main__':
    pass
