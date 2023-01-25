import requests
import pandas as pd


def send_webhook_to_ifttt(text):
    """ 파이썬 -> webhook -> ifttt -> mail 발송

    Args:
        text: 메일로 전송할 내용

    Returns: 메일은 ifttt를 통해서 처리

    """
    # 정보 가져오기
    key_df = pd.read_csv('key.csv', header=0, index_col=0)

    send_to = key_df.loc['send_to'][0]
    print(send_to)
    # 얘: sample@sample.com

    ifttt_url = key_df.loc['url'][0]
    print(ifttt_url)
    # 예: https://maker.ifttt.com/trigger/{event}/json/with/key/{my_key}

    payload = {
        "value1": send_to,
        "value2": text
        # "value3": ""
    }
    print(payload)
    requests.post(ifttt_url, json=payload)


if __name__ == '__main__':
    pass
