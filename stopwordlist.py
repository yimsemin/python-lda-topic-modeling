def get_stop_words() -> set:
    print("불용어 사전 업데이트 : 2022년 11월 2일")

    result = ["\n", "씨", "고", "수", "며", "명", "말", "전", "율"]     # 우리가 추가
    # 최초 result = ["은", '을', '를', '가', '는', '\n', '이', '도']

    # (22년 11월 2일 검색): https://www.ranks.nl/stopwords/korean
    # 뺀 것: "혼자", "우리"
    stop_words_dic = ["아", "어찌됏든", "하기보다는", "휴", "그위에", "차라리", "아이구",
                      "게다가", "하는 편이 낫다", "아이쿠", "점에서 보아", "흐흐", "아이고",
                      "비추어 보아", "놀라다", "어", "고려하면", "상대적으로 말하자면", "나",
                      "하게될것이다", "마치", "일것이다", "아니라면", "저희", "비교적",
                      "쉿", "따라", "좀", "그렇지 않으면", "의해", "보다더", "그렇지 않다면",
                      "을", "비하면", "안 그러면", "를", "시키다", "아니었다면", "에", "하게하다",
                      "하든지", "의", "할만하다", "아니면", "가", "의해서", "이라면", "으로",
                      "연이서", "좋아", "로", "이어서", "알았어", "에게", "잇따라", "하는것도",
                      "뿐이다", "뒤따라", "그만이다", "의거하여", "뒤이어", "어쩔수 없다", "근거하여",
                      "결국", "하나", "입각하여", "의지하여", "일", "기준으로", "기대여", "일반적으로",
                      "예하면", "통하여", "일단", "예를 들면", "자마자", "한켠으로는", "예를 들자면",
                      "더욱더", "오자마자", "저", "불구하고", "이렇게되면", "소인", "얼마든지",
                      "이와같다면", "소생", "마음대로", "전부", "저희", "주저하지 않고", "한마디",
                      "지말고", "곧", "한항목", "하지마", "즉시", "근거로", "하지마라", "바로",
                      "하기에", "다른", "당장", "아울러", "물론", "하자마자", "하지 않도록", "또한",
                      "밖에 안된다", "않기 위해서", "그리고", "하면된다", "이르기까지", "비길수 없다",
                      "그래", "이 되다", "해서는 안된다", "그렇지", "로 인하여", "뿐만 아니라", "요컨대",
                      "까닭으로", "만이 아니다", "다시 말하자면", "이유만으로", "만은 아니다", "바꿔 말하면",
                      "이로 인하여", "막론하고", "즉", "그래서", "관계없이", "구체적으로", "이 때문에",
                      "그치지 않다", "말하자면", "그러므로", "그러나", "시작하여", "그런 까닭에", "그런데",
                      "시초에", "알 수 있다", "하지만", "이상", "결론을 낼 수 있다", "든간에", "허",
                      "으로 인하여", "논하지 않다", "헉", "있다", "따지지 않다", "허걱", "어떤것", "설사",
                      "바와같이", "관계가 있다", "비록", "해도좋다", "관련이 있다", "더라도", "해도된다",
                      "연관되다", "아니면", "게다가", "어떤것들", "만 못하다", "더구나", "에 대해",
                      "하는 편이 낫다", "하물며", "이리하여", "불문하고", "와르르", "그리하여", "향하여",
                      "팍", "여부", "향해서", "퍽", "하기보다는", "향하다", "펄렁", "하느니", "쪽으로",
                      "동안", "하면 할수록", "틈타", "이래", "운운", "이용하여", "하고있었다", "이러이러하다",
                      "타다", "이었다", "하구나", "오르다", "에서", "하도다", "제외하고", "로부터", "다시말하면",
                      "이 외에", "까지", "다음으로", "이 밖에", "예하면", "에 있다", "하여야", "했어요",
                      "에 달려 있다", "비로소", "해요", "우리", "한다면 몰라도", "함께", "우리들", "외에도",
                      "같이", "오히려", "이곳", "더불어", "하기는한데", "여기", "마저", "어떻게", "부터",
                      "마저도", "어떻해", "기점으로", "양자", "어찌됏어", "따라서", "모두", "어때",
                      "할 생각이다", "습니다", "어째서", "하려고하다", "가까스로", "본대로", "이리하여",
                      "하려고하다", "자", "그리하여", "즈음하여", "이", "그렇게 함으로써", "다른", "이쪽",
                      "하지만", "다른 방면으로", "여기", "일때", "해봐요", "이것", "할때", "습니까",
                      "이번", "앞에서", "했어요", "이렇게말하자면", "중에서", "말할것도 없고", "이런",
                      "보는데서", "무릎쓰고", "이러한", "으로써", "개의치않고", "이와 같은", "로써",
                      "하는것만 못하다", "요만큼", "까지", "하는것이 낫다", "요만한 것", "해야한다", "매",
                      "얼마 안 되는 것", "일것이다", "매번", "이만큼", "반드시", "들", "이 정도의", "할줄알다",
                      "모", "이렇게 많은 것", "할수있다", "어느것", "이와 같다", "할수있어", "어느", "이때",
                      "임에 틀림없다", "로써", "이렇구나", "한다면", "갖고말하자면", "것과 같이", "등", "어디",
                      "끼익", "등등", "어느쪽", "삐걱", "제", "어느것", "따위", "겨우", "어느해",
                      "와 같은 사람들", "단지", "어느 년도", "부류의 사람들", "다만", "라 해도", "왜냐하면",
                      "할뿐", "언젠가", "중의하나", "딩동", "어떤것", "오직", "댕그", "어느것", "오로지",
                      "대해서", "저기", "에 한하다", "대하여", "저쪽", "하기만 하면", "대하면", "저것",
                      "도착하다", "훨씬", "그때", "까지 미치다", "얼마나", "그럼", "도달하다", "얼마만큼",
                      "그러면", "정도에 이르다", "얼마큼", "요만한걸", "할 지경이다", "남짓", "그래",
                      "결과에 이르다", "여", "그때", "관해서는", "얼마간", "저것만큼", "여러분", "약간",
                      "그저", "하고 있다", "다소", "이르기까지", "한 후", "좀", "할 줄 안다",
                      "조금", "할 힘이 있다", "자기", "다수", "너", "자기집", "몇", "너희", "자신",
                      "얼마", "당신", "우에 종합한것과같이", "지만", "어찌", "총적으로 보면",
                      "설마", "총적으로 말하면", "또한", "차라리", "총적으로", "그러나", "할지언정",
                      "대로 하다", "그렇지만", "할지라도", "으로서", "하지만", "할망정", "참", "이외에도",
                      "할지언정", "그만이다", "대해 말하자면", "구토하다", "할 따름이다", "뿐이다", "게우다",
                      "쿵", "다음에", "토하다", "탕탕", "반대로", "메쓰겁다", "쾅쾅", "반대로 말하자면",
                      "옆사람", "둥둥", "이와 반대로", "퉤", "봐", "바꾸어서 말하면", "쳇", "봐라",
                      "바꾸어서 한다면", "의거하여", "아이야", "만약", "근거하여", "아니", "그렇지않으면",
                      "의해", "와아", "까악", "따라", "응", "툭", "힘입어", "아이", "딱", "그",
                      "참나", "삐걱거리다", "다음", "년", "보드득", "버금", "월", "비걱거리다",
                      "두번째로", "일", "꽈당", "기타", "령", "응당", "첫번째로", "영", "해야한다",
                      "나머지는", "에 가서", "그중에서", "이", "각", "견지에서", "삼", "각각",
                      "형식으로 쓰여", "사", "여러분", "입장에서", "오", "각종", "위해서", "육", "각자",
                      "단지", "륙", "제각기", "의해되다", "칠", "하도록하다", "하도록시키다", "팔", "와",
                      "뿐만아니라", "구", "과", "반대로", "이천육", "그러므로", "전후", "이천칠", "그래서",
                      "전자", "이천팔", "고로", "앞의것", "이천구", "한 까닭에", "잠시", "하나", "하기 때문에",
                      "잠깐", "둘", "거니와", "하면서", "셋", "이지만", "그렇지만", "넷", "대하여", "다음에",
                      "다섯", "관하여", "그러한즉", "여섯", "관한", "그런즉", "일곱", "과연", "남들", "여덟",
                      "실로", "아무거나", "아홉", "아니나다를가", "어찌하든지", "령", "생각한대로", "같다", "영",
                      "진짜로", "비슷하다", "한적이있다", "예컨대", "하곤하였다", "이럴정도로", "하", "어떻게",
                      "하하", "만약", "허허", "만일", "아하", "위에서 서술한바와같이", "거바", "인 듯하다",
                      "와", "하지 않는다면", "오", "만약에", "왜", "무엇", "어째서", "무슨", "무엇때문에",
                      "어느", "어찌", "어떤", "하겠는가", "아래윗", "무슨", "조차", "어디", "한데", "어느곳",
                      "그럼에도 불구하고", "더군다나", "여전히", "심지어", "더욱이는", "까지도", "어느때",
                      "조차도", "언제", "하지 않도록", "야", "않기 위하여", "이봐", "때", "어이", "시각",
                      "여보시오", "무렵", "흐흐", "시간", "흥", "동안", "휴", "어때", "헉헉", "어떠한",
                      "헐떡헐떡", "하여금", "영차", "네", "여차", "예", "어기여차", "우선", "끙끙", "누구",
                      "아야", "누가 알겠는가", "앗", "아무도", "아야", "줄은모른다", "콸콸", "줄은 몰랏다",
                      "졸졸", "하는 김에", "좍좍", "겸사겸사", "뚝뚝", "하는바", "주룩주룩", "그런 까닭에",
                      "솨", "한 이유는", "우르르", "그러니", "그래도", "그러니까", "또", "때문에", "그리고",
                      "그", "바꾸어말하면", "너희", "바꾸어말하자면", "그들", "혹은", "너희들", "혹시", "타인",
                      "답다", "것", "및", "것들", "그에 따르는", "너", "때가 되어", "위하여", "즉", "공동으로",
                      "지든지", "동시에", "설령", "하기 위하여", "가령", "어찌하여", "하더라도", "무엇때문에",
                      "할지라도", "붕붕", "일지라도", "윙윙", "지든지", "나", "몇", "우리", "거의", "엉엉",
                      "하마터면", "휘익", "인젠", "윙윙", "이젠", "오호", "된바에야", "아하", "된이상",
                      "어쨋든", "만큼", "만 못하다"]

    result.extend(stop_words_dic)

    return set(result)

"""
def loadstopwordlist():  # 함수: 불용어 사전 불러오기
    try:
        from stopwordlist import set_stop_words     # 디렉토리 내 'stopwordlist.py'에서 불용어 사전을 가져옴
        stop_words = set_stop_words()
    except FileNotFoundError:                       # 없으면 아래 불용어 리스트 사용
        print("불용어사전 파일(stopwordlist.py)가 같은 디렉토리 내 찾을 수 없습니다. 기본 불용어 사전을 사용합니다.\n")
        stop_words = ["은", '을', '를', '가', '는', '\n', '이', '도']

    return stop_words
"""

if __name__ == '__main__':
    pass
