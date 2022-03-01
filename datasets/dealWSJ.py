# _*_coding: utf-8_*_

import json
import os

root_dir = "./data/penntree"
import StanfordDependencies
from multiprocessing import Pool

sd = StanfordDependencies.get_instance(backend='subprocess')


def parser(line):
    try:
        sent = sd.convert_tree(line)
        text, text_pos, text_dp = [], [], []
        for token in sent:
            text.append(token.form.lower())
            text_pos.append(token.pos)
            text_dp.append(token.head)
        return text, text_pos, text_dp
    except:
        return [], [], []


def posparser(file):
    article, pos, dp = [], [], []
    print(file)
    with open(file, "r", encoding="utf-8") as f:
        data = [x for x in f.readlines()]
        with Pool(processes=16) as p:
            for text, text_pos, text_dp in p.imap(parser, data):
                if len(text) == 0: continue
                assert len(text) == len(text_pos) and len(text) == len(text_dp)
                article.append(text)
                pos.append(text_pos)
                dp.append(text_dp)
    return article, pos, dp


def getdata(path):
    data_raw = []
    for file_id in path:
        path_list = sorted([os.path.join(file_id, i) for i in os.listdir(file_id)])
        for file in path_list:
            article, pos, dp = posparser(file)
            for text, text_pos, text_dp in zip(article, pos, dp):
                assert len(text) == len(text_pos) and len(text) == len(text_dp)
                data_raw.append({
                    "text": [text],
                    "pos": [text_pos],
                    "head": [text_dp]
                })
    return data_raw


def savedata(data_raw, mode_name):
    save_path = mode_name + "-wsj.json"
    with open(save_path, "a+", encoding="utf-8") as f:
        for text in data_raw:
            json.dump(text, f, ensure_ascii=False)
            f.write("\n")


if __name__ == '__main__':
    train_id = [str(i) if len(str(i)) == 2 else '0' + str(i) for i in range(19)]
    valid_id = [str(i) if len(str(i)) == 2 else '0' + str(i) for i in range(19, 22)]
    test_id = [str(i) if len(str(i)) == 2 else '0' + str(i) for i in range(22, 25)]
    
    train_dir = [os.path.join(root_dir, i) for i in train_id]
    valid_dir = [os.path.join(root_dir, i) for i in valid_id]
    test_dir = [os.path.join(root_dir, i) for i in test_id]
    
    data_raw = getdata(train_dir)
    savedata(data_raw, "train")
    data_raw = getdata(valid_dir)
    savedata(data_raw, "valid")
    data_raw = getdata(test_dir)
    savedata(data_raw, "test")

