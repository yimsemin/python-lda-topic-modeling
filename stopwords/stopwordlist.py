def get_stop_words():
    """ return set(stopword list)

    불용어 사전 불러오기(stopwordlist.txt)
    파일이 없으면 기본 불용어 사전을 사용

    Returns:
        set(stopwordlist.txt)set 형식의 불용어 사전

    """
    try:
        with open('stopwordlist.txt', 'r') as f:
            stop_word_list = f.read().splitlines()
        print(stop_word_list[0] if '#' in stop_word_list[0] else None)  # 불용어 사전 정보 출력

        i = 0                                                           # 주석 제거
        while '#' in stop_word_list[i]:
            stop_word_list[i] = ''
            i += 1

    except FileNotFoundError:
        stop_word_list = ['\n', '을', '를', '은', '가', '는', '이', '도', '수', '며', '고']      # 기본 불용어

    return set(stop_word_list).remove('')


if __name__ == '__main__':
    pass
