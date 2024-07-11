import re
import spacy
import random

def read_file_lines(file_path):
    file_en, file_de = [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line == 'English:' or line == 'German:':
                if len(cur_str) > 0:
                    temp_str = cur_str.strip()
                    splited = re.split('[.?!]', temp_str)
                    parts = [s for s in splited if len(s) > 0]
                    cur_list.extend(parts)
                    cur_str = ''
                if line == 'English:':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue
            cur_str += line + ' '
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_en, file_de


def flatten(data):
    res = []
    for lang in data:
        new = []
        for sen in lang:
            splited = re.split('[.?!]', sen)
            parts = [s for s in splited if len(s) > 0]
            if len(parts) > 0:
                new.extend(parts)
        res.append(new)
    return tuple((res[0], res[1]))


from tqdm import tqdm


def add_dp(data):
    en, ger = data
    # path = "en_core_web_md"
    path = "en_core_web_sm"
    parser = spacy.load(path)
    res_ger = []

    for e, g in tqdm(zip(en, ger)):
        roots = []
        mods = []
        res = parser(e)
        for token in res.sents:
            roots.append(token.root)
            m = [s for s in list(token.root.children) if len(s) > 1]
            if len(m) >= 2:
                m = random.choices(m, k=2)
            mods.append(tuple(m))
            # if token.dep_ == 'ROOT':
            #     roots += token.text
            #     roots += ', '
            #     m = '('
            #     for t in res:
            #         if t.head.text == token.text and t.text != token.text and t.text not in ['.', ',', '!', "'",                                                                   ')', '(', '?']:
            #             m += t.text
            #             m += ', '
            #     m = m[:len(m) - 2]
            #     m += ')'
            #     mods += m
            #     mods += ','

        roots = str(roots).translate({ord(i): None for i in '[]'})
        mods = str(mods).translate({ord(i): None for i in '[]'})

        g += ' Roots in English: ' + roots + ' Modifiers in English ' + mods
        # g += ' Roots in English: ' + roots
        # g += ' '
        # g += ' Modifiers in English ' + mods
        res_ger.append(g)

    return en, res_ger
