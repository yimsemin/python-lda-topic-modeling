""" 전처리 결과 문자열을 토큰 리스트로 변환하는 함수
"""
import pandas as pd


def parse_tokenized_article(line):
    # 전처리 결과가 빈 셀이면 엑셀에서 NaN으로 읽히므로 빈 문서로 처리한다.
    if isinstance(line, (list, tuple)):
        return [word for word in line if word]

    try:
        if line is None or pd.isna(line):
            return []
    except (TypeError, ValueError):
        pass

    return [word.strip() for word in str(line).split(',') if word.strip()]
