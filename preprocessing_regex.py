import re
from itertools import islice

import util.openxlsx as openxlsx


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


if __name__ == '__main__':
    article_list = openxlsx.load_series_from_xlsx('input/6-12article.xlsx',
                                                  'article',
                                                  sheet_name='preprocessed_result',
                                                  is_list_in_list=True)
    MY_RULE = _my_rule()

    new_article_list = [multiple_replace(MY_RULE, ' '.join(article)) for article in article_list]

    openxlsx.save_to_xlsx(new_article_list, 'input/6-12article.xlsx', 'article', 'preprocessed_result')
