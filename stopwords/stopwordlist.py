def get_stop_words(stopwordlist_location: str) -> set:
    """ 저장된 경로의 txt 불용어 사전 불러오기, 파일이 없으면 기본 불용어 사전 사용

    불용어 사전 파일은 # 표시로 주석을 표시할 수 있음
    불용어는 엔터로 구분하기

    Args:
        stopwordlist_location(str): 불용어 사전 위치(예: 'stopwords/stopwordlist.txt')

    Returns:
        set 형식의 불용어 사전

    """
    try:
        with open(stopwordlist_location, 'r') as f:
            print('-- 저장된 불용어 사전을 불러옵니다.')
            txt_lines = f.read().splitlines()

        comments = [line for line in txt_lines if '#' in line]
        if not comments:
            print('---- 주석 없음')
        else:
            [print('---- '+str(i)) for i in comments]       # 불용어 사전의 코멘트 출력

        stop_word_list = [line for line in txt_lines if '#' not in line]

    except FileNotFoundError:
        print('-- 기본 불용어 사전을 사용합니다.')
        stop_word_list = ['\n', '을', '를', '은', '가', '는', '이', '도', '수', '며', '고']      # 기본 불용어

    print('-- 불용어 사전 예시 : '+', '.join(stop_word_list[0:4])+' ...')

    return set(stop_word_list)


def main():
    get_stop_words('stopwordlist.txt')


if __name__ == '__main__':
    main()
