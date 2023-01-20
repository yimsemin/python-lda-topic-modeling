import zipfile
import pandas as pd


def save_to_xlsx(data_to_save,
                 save_to_this_file: str = 'test.xlsx',
                 column_name: str = 'article',
                 sheet_name: str = 'result'):
    if save_to_this_file.split('.')[-1] != 'xlsx':
        raise ValueError("데이터를 저장할 파일이 엑셀파일이 아닙니다.")

    if type(data_to_save) not in [list, pd.Series, pd.DataFrame]:
        raise ValueError("저장할 데이터의 형식은 Series, DataFrame, list 중 하나여야 합니다.")

    if type(data_to_save[0]) is list:
        # 리스트 속에 리스트 구조로 되어있는지? 우선은 첫번째 요소만 확인함
        try:
            print("리스트 속 리스트를 띄어쓰기로 분해하여 저장합니다.")
            data_to_save = pd.Series(data_to_save).map(lambda word: ' '.join(word))
        except ValueError:
            pass

    df_to_save = pd.DataFrame()
    df_to_save[column_name] = data_to_save

    try:
        # 데이터를 저장할 파일이 있는 경우 -> 파일에 시트 추가/덮어쓰기
        with pd.ExcelWriter(save_to_this_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df_to_save.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        # 데이터를 저장할 파일이 없는 경우 -> 파일 새로 만들기
        with pd.ExcelWriter(save_to_this_file, mode='w', engine='openpyxl') as writer:
            df_to_save.to_excel(writer, sheet_name=sheet_name, index=False)
    except zipfile.BadZipfile:
        print('엑셀파일을 지우고 다시 시도해보세요.')


def load_series_from_xlsx(load_from_this_file: str = 'test.xlsx',
                          column_name: str = 'article',
                          sheet_name=None,
                          is_list_in_list: bool = False) -> pd.Series:
    if sheet_name is None:
        sheet_name = 0

    f = pd.read_excel(load_from_this_file, sheet_name=sheet_name)
    loaded_data = f[column_name].dropna()

    if is_list_in_list is True:
        # 리스트 속 리스트 구조로 가져오기
        loaded_data = loaded_data.map(lambda line: line.split())

    return loaded_data


def load_dictionary_from_xlsx(load_from_this_file: str = 'test.xlsx',
                              column_value_name: str = 'VALUE',
                              sheet_name=0,
                              fill_na='') -> dict:
    # 중간에 공백줄 지우는 것 구현 필요함 nan: ''으로 사이에 끼어져있음

    df = pd.read_excel(load_from_this_file, sheet_name=sheet_name, index_col=0).fillna(fill_na)
    df = df.to_dict()
    loaded_data = df[column_value_name]

    return loaded_data


def load_df_from_xlsx():
    pass


if __name__ == '__main__':
    pass
