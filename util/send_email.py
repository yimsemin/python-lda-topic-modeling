""" 파이썬 코드로 이메일 보내기 (IFTTT 설정 필요)

IFTTT 셋팅:
    IF Receive a web request
        Event Name: {event}
    Then Send an email(gmail)
        Gmail account: 각자 계정
        to address: {{Value1}}
        Subject: [파이썬 봇] {{Value2}}
        Body: 파이썬 코드에 따라 메일이 전송되었습니다.
              작업: {{Value2}}
              시기: {{OccurredAt}}
              내용: {{EventName}}
csv 셋팅:
    같은 폴더 내 key.csv 생성
    아래의 구조로 내용 작성
               , value
        send_to, {여기에 보낼 대상 이메일}            # 예: sample@sample.com
        url    , {여기에 ifttt webhook url 링크}   # 예: https://maker.ifttt.com/trigger/{event}/with/key/{my_key}
                                                # 주의: ~~/json/with/key~~ 가 아니라 ~~/with/key~~ 임
                                                #     json이 안되서 'application/x-www-form-urlencoded'를 사용함
"""

import pandas as pd
import requests


def send_email_via_ifttt(text: str, send_to='sample@sample.com'):
    """ 파이썬에서 webhook 전송 -> ifttt에서 email 발송

    메일 전송은 ifttt에서 진행
    ifttt webhook url은 동일 디렉토리 key.csv에서 가져옴(url, {여기에 주소 작성})

    Args:
        text(str): 메일로 전송할 내용, 한 문장만 아주 짧게 특수문자 없이 작성
        send_to(str, optional): 메일을 전송할 이메일 주소, 공백의 경우 key.csv에서 가져옴(send_to, {여기에 보낼 대상 이메일})

    """
    try:
        key_df = pd.read_csv('key.csv', header=0, index_col=0)
        if send_to == 'sample@sample.com':
            send_to = key_df.loc['send_to'][0]
        ifttt_url = key_df.loc['url'][0]
    except FileNotFoundError as e1:
        print(e1, 'key.csv 파일이 없습니다.')
        raise
    except ValueError as e2:
        print(e2, 'key.csv 내용을 확인해주세요.')
        raise

    headers = {'Content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(ifttt_url + f'?value1={send_to}&value2={text}', headers=headers)

    if response.status_code == 200:
        print(f'[IFTTT] {text}의 내용으로 {send_to}에게 이메일을 전송합니다.')
    else:
        print('[IFTTT] 에러: ', response.status_code, response.text.strip('\n'))


if __name__ == '__main__':
    send_email_via_ifttt('test_테스트')
