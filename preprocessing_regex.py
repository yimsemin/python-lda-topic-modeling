import re
from itertools import islice

import util.openxlsx as openxlsx


def _setting():
    # input
    input_data = {
        'xlsx_name': 'input/test.xlsx',
        'sheet_name': 'preprocessed',
        'column_name': 'article',
    }

    # output
    output_data = {
        'result_xlsx_name': 'input/test_preprocessed.xlsx',
        'result_sheet_name': 'preprocessed',
        'result_column_name': 'article'
    }

    article_series = openxlsx.load_series_from_xlsx(input_data['xlsx_name'],
                                                    input_data['column_name'],
                                                    input_data['sheet_name'],
                                                    is_list_in_list=True)

    return input_data, output_data, article_series


def _my_rule():
    rule_dict = {
        '대해': ' ', '관련': ' ', '대한': ' ', '위해': ' ', '통해': ' ', '라며': ' ', '기자': ' ',
        '정도': ' ', '때문': ' ', '어치': ' ', '원래': ' ', '동아닷컴': ' ',
        '확\\s진자': '확진자', '\\s진자\\s': '확진자', '프로\\s듀스': '프로듀스'
    }

    return rule_dict


def multiple_simple_replace(rule_dict, text):
    # Create a regular expression from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, rule_dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: rule_dict[mo.string[mo.start():mo.end()]], text)


def multiple_replace(rule_dict, text):
    # https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex

    groups_no = [re.compile(pattern).groups for pattern in rule_dict]

    def repl_func(m):
        all_groups = m.groups()

        # Use 'i' as the index within 'all_groups' and 'j' as the main
        # group index.
        i, j = 0, 0

        while i < len(all_groups) and all_groups[i] is None:
            # Skip the inner groups and move on to the next group.
            i += (groups_no[j] + 1)

            # Advance the main group index.
            j += 1

        # Extract the pattern and replacement at the j-th position.
        pattern, repl = next(islice(rule_dict.items(), j, j + 1))

        return re.sub(pattern, repl, all_groups[i])

    # Create the full pattern using the keys of 'repl_dict'.
    full_pattern = '|'.join(f'({pattern})' for pattern in rule_dict)

    return re.sub(full_pattern, repl_func, text)


class TextListRegexEditor:
    def __init__(self):
        self.textlist = []
        self.rule = {}

    def load_textlist(self,
                      xlsx_name: str,
                      column_name='article',
                      sheet_name: str = 'preprocessed_result',
                      is_list_in_list: bool = True):
        self.textlist = openxlsx.load_series_from_xlsx(xlsx_name,
                                                       column_name,
                                                       sheet_name=sheet_name,
                                                       is_list_in_list=is_list_in_list)

    def load_rule(self,
                  xlsx_name: str = 'regexrule.xlsx',
                  sheet_name: str = 0):
        # Key: 첫번째 열
        # value: 두번째 열
        self.rule = openxlsx.load_dictionary_from_xlsx(xlsx_name, 'VALUE', sheet_name=sheet_name)
        # 엑셀에 있는 값에 \s 를 \\s 이런식으로 가져오는데 확인 필요

    def load_default_rule(self):
        self.rule = _my_rule()

    def replace_with_rule(self, rule_dict):
        pass


def main():
    # setting
    _, output_data, article_series = _setting()
    my_rule = _my_rule()

    # preprocess with regex rule
    new_article_series = [multiple_replace(my_rule, ' '.join(article)) for article in article_series]

    # save result
    openxlsx.save_to_xlsx(new_article_series,
                          output_data['result_xlsx_name'],
                          output_data['result_column_name'],
                          output_data['result_sheet_name'])


if __name__ == '__main__':
    main()
