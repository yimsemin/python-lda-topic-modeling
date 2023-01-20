import os
import shutil
import zipfile
import tempfile

from datetime import datetime


def _setting():
    # okt 사전이 위치한 경로
    okt_location = "/opt/homebrew/Caskroom/miniconda/base/envs/iedu-lda/lib/python3.10/site-packages/konlpy/java"

    custom_noun_txt = 'my_nouns.txt'     # 커스텀 명사 사전 (1줄에 "단어" 1개씩 / 따옴표는 빼고)
    custom_typos_txt = 'my_typos.txt'    # 커스텀 타이포 사전 (1줄에 "오타인단어 올바른단어"로 띄어쓰기 구분해서 작성 / 따옴표는 빼고)

    # 각 커스텀 사전의 정렬 수정
    for txt in [custom_noun_txt, custom_typos_txt]:
        with open(txt, 'r') as f:
            lines = f.read().splitlines()
            sorted_lines = [line + '\n' for line in sorted(lines)]
        with open(txt, 'w') as f:
            f.writelines(sorted_lines)

    return okt_location, custom_noun_txt, custom_typos_txt


def find_latest_okt(okt_location: str):
    latest_okt = [f_name for f_name in os.listdir(okt_location) if 'open-korean-text' in f_name]

    latest_okt_name = latest_okt[-1]
    latest_okt_dir = okt_location + '/' + latest_okt_name

    return latest_okt_name, latest_okt_dir


def remove_from_zip(zip_f_name, *filenames):
    # https://stackoverflow.com/questions/4653768/overwriting-file-in-ziparchive
    # zipfile 모듈에서는 기존 파일 업데이트가 안되서 우선 삭제하는 것을 추가함
    tempdir = tempfile.mkdtemp()
    try:
        temp_name = os.path.join(tempdir, 'new.zip')
        with zipfile.ZipFile(zip_f_name, 'r') as zip_read:
            with zipfile.ZipFile(temp_name, 'w') as zip_write:
                for item in zip_read.infolist():
                    if item.filename not in filenames:
                        data = zip_read.read(item.filename)
                        zip_write.writestr(item, data)
        shutil.move(temp_name, zip_f_name)
    finally:
        shutil.rmtree(tempdir)


def add_custom_dict_to_okt_in_workdir(latest_okt_name: str,
                                      latest_okt_dir: str,
                                      custom_noun_txt: str,
                                      custom_typos_txt: str):

    # 작업폴더로 okt 가져오기
    shutil.copyfile(latest_okt_dir, latest_okt_name)

    # 기존 okt 백업
    # https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
    backup_counter = 1
    backup_okt = latest_okt_name + '.backup_' + datetime.today().strftime("%Y%m%d") + "{}"
    while os.path.isfile(backup_okt.format(backup_counter)):
        backup_counter += 1
    shutil.copyfile(latest_okt_dir, backup_okt.format(backup_counter))
    print('기존 okt 사전이 백업됨')

    # 압축파일에 사전 추가할 준비
    custom_noun_txt_destination = '/org/openkoreantext/processor/util/noun/my_nouns.txt'
    custom_typos_txt_destination = '/org/openkoreantext/processor/util/typos/my_typos.txt'

    # 혹시 이전에 커스텀 사전을 업로드한 적이 있을 경우, 사전을 삭제함
    remove_from_zip(latest_okt_name, custom_noun_txt_destination)
    remove_from_zip(latest_okt_name, custom_typos_txt_destination)

    # 새로운 사전을 추가함
    with zipfile.ZipFile(latest_okt_name, 'a') as myzip:
        myzip.write(custom_noun_txt, arcname=custom_noun_txt_destination)
        print('custom noun 사전을 업데이트 함')
        myzip.write(custom_typos_txt, arcname=custom_typos_txt_destination)
        print('custom typos 사전을 업데이트 함')


def apply_custom_okt(latest_okt_name, latest_okt_dir):
    os.replace(latest_okt_name, latest_okt_dir)


def main():
    okt_location, custom_noun_txt, custom_typos_txt = _setting()
    latest_okt_name, latest_okt_dir = find_latest_okt(okt_location)
    add_custom_dict_to_okt_in_workdir(latest_okt_name, latest_okt_dir, custom_noun_txt, custom_typos_txt)
    apply_custom_okt(latest_okt_name, latest_okt_dir)


if __name__ == '__main__':
    main()
