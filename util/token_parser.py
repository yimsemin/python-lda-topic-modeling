""" 전처리 결과 문자열을 토큰 리스트로 변환하는 함수
"""
import numbers

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


def parse_tokenized_series(series):
    source_series = pd.Series(series)
    tokenized_series = source_series.map(parse_tokenized_article)
    tokenized_series.attrs['source_series'] = source_series

    return tokenized_series


def get_empty_document_indices(tokenized_article_series):
    tokenized_article_series = pd.Series(tokenized_article_series).map(parse_tokenized_article)

    return tokenized_article_series[tokenized_article_series.map(len) == 0].index.tolist()


def _get_source_preview(source_series, index, preview_length: int) -> str:
    if source_series is None:
        return ''

    source_series = pd.Series(source_series)
    if index in source_series.index:
        value = source_series.loc[index]
    elif isinstance(index, numbers.Integral) and 0 <= index < len(source_series):
        value = source_series.iloc[index]
    else:
        return ''

    try:
        if value is None or bool(pd.isna(value)):
            return ''
    except (TypeError, ValueError):
        pass

    preview = str(value).replace('\r', ' ').replace('\n', ' ').strip()
    if len(preview) > preview_length:
        return preview[:preview_length]+'...'

    return preview


def log_empty_documents(tokenized_article_series, label: str, detail_message: str = None,
                        source_series=None, preview_length: int = 10) -> int:
    empty_indices = get_empty_document_indices(tokenized_article_series)
    print(f'-- {label} 빈 문서: {len(empty_indices)}건')

    if empty_indices:
        if source_series is None:
            source_series = getattr(tokenized_article_series, 'attrs', {}).get('source_series')

        print('-- 빈 문서 번호 기준: pandas Series index 기준입니다. 0부터 시작하며 엑셀 화면 행 번호로 변환하지 않습니다.')
        for index in empty_indices:
            preview = _get_source_preview(source_series, index, preview_length)
            print(f'---- 문서 {index}: 원문 앞 {preview_length}글자 "{preview}"')
        if detail_message is not None:
            print(detail_message)

    return len(empty_indices)
