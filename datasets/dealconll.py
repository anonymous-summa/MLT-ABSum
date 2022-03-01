# _*_coding: utf-8_*_

import json

import pyconll

total_pos = []
total_relation = []

for file in ["train","valid","test"]:
    data = pyconll.load_from_file(file + ".conll")
    article = []
    for sents in data:
        text, head, pos, relations = [], [], [], []
        for token in sents:
            text.append(token._form.lower())
            head.append(int(token.head))
            pos.append(token.xpos)
            relations.append(token.deprel)

            total_pos.append(token.xpos)
            total_relation.append(token.deprel)

        article.append({
            "text": [text],
            "pos": [pos],
            "head": [head],
            "relation":[relations]
            }
        )
    
    with open(file + "-conll.json", "w+", encoding="utf-8") as f:
        for text in article:
            json.dump(text, f, ensure_ascii=False)
            f.write("\n")

total_pos = list(set(total_pos))
total_relation = list(set(total_relation))

posid = {k:i for i,k in enumerate(total_pos)}
relation2id = {k:i for i,k in enumerate(total_relation)}
print(posid)
print(relation2id)
