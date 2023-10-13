import re


def substitute_show_to_json(string):
    return re.sub(r'(\w+)\.show\(\)', r'print(\1.to_json())', string)
