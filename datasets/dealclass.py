# _*_coding: utf-8_*_

import json
import random
from multiprocessing.pool import Pool
import multiprocessing

cpu_core = multiprocessing.cpu_count()

import spacy

nlp = spacy.load('en_core_web_sm')


def parser(article):
    text = article[0].strip().lower()
    doc = nlp(text)
    return {
        "text": [(" ".join([str(x) for x in sent])).strip().split(" ") for sent in doc.sents],
        "class": article[1].lower().strip()
    }


def get_ag_data():
    for file in ["train", "test"]:
        text_path = file + "_texts.txt"
        text_lable = file + "_labels.txt"
        data = []
        with open(text_path, "r", encoding="utf-8") as f1, open(text_lable, "r", encoding="utf-8") as f2:
            article = [(text.strip(), label.strip()) for text, label in zip(f1.readlines(), f2.readlines())]
            print(len(article))
            with Pool(processes=cpu_core) as p:
                for item in p.imap(parser, article):
                    data.append(item)
        print(len(data))
        if file == "train":
            random.shuffle(data)
            all_len = int(len(data) * 0.95)
            train_data = data[:all_len]
            valid_data = data[all_len:]
            savedata(train_data, "train")
            savedata(valid_data, "valid")
        else:
            savedata(data, "test")


def savedata(data_raw, mode_name):
    save_path = mode_name + "-ag.json"
    with open(save_path, "w+", encoding="utf-8") as f:
        for text in data_raw:
            json.dump(text, f, ensure_ascii=False)
            f.write("\n")


if __name__ == '__main__':
    get_ag_data()

