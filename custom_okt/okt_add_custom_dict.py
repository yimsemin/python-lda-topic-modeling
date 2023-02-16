import os
import shutil
import zipfile
import tempfile

from datetime import datetime


def _setting():
    # okt 사전이 위치한 경로
    # 각자 설치 장소가 다를 수 있음
    okt_location = "/opt/homebrew/Caskroom/miniconda/base/envs/iedu-lda/lib/python3.10/site-packages/konlpy/java"

    custom_noun_txt = 'my_nouns.txt'     # 커스텀 명사 사전 (1줄에 단어 1개씩)
    custom_typos_txt = 'my_typos.txt'    # 커스텀 타이포 사전 (1줄에 오타인단어 올바른단어 순서로 작성 / 띄어쓰기로 구분)

    return okt_location, custom_noun_txt, custom_typos_txt


def arrange_custom_okt(custom_noun_txt: str, custom_typos_txt: str):
    # 각 커스텀 사전의 중복 제거 + 정렬 (덮어쓰기)
    for txt in [custom_noun_txt, custom_typos_txt]:
        with open(txt, 'r') as f:
            lines = f.read().splitlines()
            result = [line + '\n' for line in sorted(list(set(lines)))]
        with open(txt, 'w') as f:
            f.writelines(result)


def find_latest_okt(okt_location: str):
    latest_okt = [f_name for f_name in os.listdir(okt_location) if 'open-korean-text' in f_name]
    if not latest_okt:
        print('open-korean-text-x.x.x.jar 파일을 찾을 수 없습니다. 경로를 확인해주세요.')
        latest_okt_name = None
        latest_okt_dir = None
    else:
        latest_okt_name = latest_okt[-1]
        print('-- okt파일 확인 : '+latest_okt_name)

        latest_okt_dir = okt_location + '/' + latest_okt_name

    return latest_okt_name, latest_okt_dir


def backup_original_okt_to_workdir(latest_okt_name: str, latest_okt_dir: str):
    # 현재 폴더에 백업
    # 백업시점에 따른 파일명이 설정됨 (예: open-korean-text-2.1.0.jar.backup_yyyymmdd_0)
    # https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
    backup_counter = 1
    backup_okt = latest_okt_name + '.backup_' + datetime.today().strftime("%Y%m%d") + "_{}"
    while os.path.isfile(backup_okt.format(backup_counter)):
        backup_counter += 1
    shutil.copyfile(latest_okt_dir, backup_okt.format(backup_counter))
    print('기존 okt 사전이 백업됨 : ' + str(backup_okt.format(backup_counter)))


def remove_from_zip(zip_f_name: str, *filenames: str):
    # zipfile 모듈에서는 압축 파일 안의 파일 업데이트가 안됨 -> 우선 zip 파일에서 해당 파일 삭제하는 함수를 추가함
    # https://stackoverflow.com/questions/4653768/overwriting-file-in-ziparchive
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


def add_custom_dict_to_okt(latest_okt_dir: str, custom_noun_txt: str, custom_typos_txt: str):
    # custom_dict 경로 설정
    custom_noun_txt_destination = 'org/openkoreantext/processor/util/noun/my_nouns.txt'
    custom_typos_txt_destination = 'org/openkoreantext/processor/util/typos/my_typos.txt'

    # 이전에 커스텀 사전을 추가한 경우, jar에 있는 커스텀 사전을 삭제함
    need_to_update = False
    with zipfile.ZipFile(latest_okt_dir, 'r') as myzip:
        if custom_noun_txt_destination in myzip.namelist():
            print('-- 기존 my_nouns.txt 파일을 교체합니다.')
            need_to_update = True
        if custom_typos_txt_destination in myzip.namelist():
            print('-- 기존 my_typos.txt 파일을 교체합니다.')
            need_to_update = True

    if need_to_update:
        remove_from_zip(latest_okt_dir, custom_noun_txt_destination, custom_typos_txt_destination)
    else:
        print('-- 교체할 파일이 없습니다.')

    # 새로운 사전을 추가함
    with zipfile.ZipFile(latest_okt_dir, 'a') as myzip:
        myzip.write(custom_noun_txt, arcname=custom_noun_txt_destination)
        print('-- custom noun 사전을 업데이트 함')
        myzip.write(custom_typos_txt, arcname=custom_typos_txt_destination)
        print('-- custom typos 사전을 업데이트 함')


def check_if_custom_okt():  # todo
    pass


def main():
    okt_location, custom_noun_txt, custom_typos_txt = _setting()
    latest_okt_name, latest_okt_dir = find_latest_okt(okt_location)
    backup_original_okt_to_workdir(latest_okt_name, latest_okt_dir)
    add_custom_dict_to_okt(latest_okt_dir, custom_noun_txt, custom_typos_txt)


if __name__ == '__main__':
    main()
