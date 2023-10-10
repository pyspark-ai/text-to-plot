import re


def substitute_show_to_json(string_list):
    return [re.sub(r'(\w+)\.show\(\)', r'print(\1.to_json())', string) for string in string_list]
