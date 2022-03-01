# _*_coding: utf-8_*_

import json
import os
import sys

from multiprocessing.pool import Pool
import multiprocessing


cpu_core = multiprocessing.cpu_count()

word2id = json.load(open("word2id.json", "r"))
id2word = json.load(open("id2word.json", "r"))
print(len(word2id))

pos2id = {'#': 0, '$': 1, "''": 2, '(': 3, ')': 4, ',': 5, '-LRB-': 6, '-RRB-': 7, '.': 8, ':': 9, 'AFX': 10, 'CC': 11,'CD': 12, 'DT': 13, 'EX': 14, 'FW': 15, 'HYPH': 16, 'IN': 17, 'JJ': 18, 'JJR': 19, 'JJS': 20, 'LS': 21, 'MD': 22, 'NFP': 23, 'NN': 24, 'NNP': 25, 'NNPS': 26, 'NNS': 27, 'PDT': 28, 'POS': 29, 'PRP': 30, 'PRP$': 31, 'RB': 32, 'RBR': 33, 'RBS': 34, 'RP': 35, 'SYM': 36, 'TO': 37, 'UH': 38, 'VB': 39, 'VBD': 40, 'VBG': 41,'VBN': 42, 'VBP': 43, 'VBZ': 44, 'WDT': 45, 'WP': 46, 'WP$': 47, 'WRB': 48, '``': 49}

class2id = {"world": 0,
            "sports": 1,
            "business": 2,
            "sci/tech": 3}

relation2id = {'neg': 0, 'advcl': 1, 'prt': 2, 'mwe': 3, 'xcomp': 4, 'csubjpass': 5, 'mark': 6, 'rel': 7, 'dobj': 8, 'nsubjpass': 9, 'ROOT': 10, 'adpobj': 11, 'rcmod': 12, 'adp': 13, 'nsubj': 14, 'acomp': 15, 'infmod': 16, 'conj': 17, 'amod': 18, 'iobj': 19, 'num': 20, 'cc': 21, 'compmod': 22, 'auxpass': 23, 'partmod': 24, 'dep': 25, 'p': 26, 'ccomp': 27, 'aux': 28, 'adpmod': 29, 'advmod': 30, 'parataxis': 31, 'nmod': 32, 'attr': 33, 'poss': 34, 'csubj': 35, 'cop': 36, 'det': 37, 'adpcomp': 38, 'expl': 39, 'appos': 40
}

def convert2id(text):
    return [word2id.get(w, word2id['UNK']) for w in text]

def convertPos2id(pos_text):
    return [pos2id.get(w) for w in pos_text]


def conevertClass2id(class_text):
    return class2id.get(class_text)

def convertRelation2id(relation_text):
    return [relation2id.get(w) for w in relation2id]


def parserArticle(article):
    text = json.loads(article)
    doc, abstract = [], []
    oovs = []
    enc_extend_words = []
    for sent in text["text"]:
        w2id = convert2id(sent)
        extend_words = []
        for w, x in zip(sent, w2id):
            if x == word2id["UNK"]:
                if w not in oovs:
                    oovs.append(w)
                extend_words.append(len(word2id)+oovs.index(w))
            else:
                extend_words.append(x)
        
        doc.append(w2id)
        enc_extend_words.append(extend_words)
    
    dec_extend_words = []
    for ref in text["abstract"]:
        w2id = convert2id(ref)
        extend_words = []
        for w, x in zip(ref, w2id):
            if x == word2id["UNK"]:
                if w not in oovs:
                    oovs.append(w)
                extend_words.append(len(word2id)+oovs.index(w))
            else:
                extend_words.append(x)
                
        abstract.append(w2id)
        dec_extend_words.append(extend_words)


    return {
        "text": doc,
        "enc_extend_words":  enc_extend_words,
        "abstract": abstract,
        "doc_extend_words": dec_extend_words,
        "oovs": oovs
    }


def convertTextSummarization(dataset="cnndm"):
    print(dataset, " is converting  words to nums .....")
    root1 = "./datasets"
    count_unknow_words = 0
    total_words = 0

    for file in ["train", "valid", "test"]:
        file_path = os.path.join(root1 + "/" + dataset, file + "-" + dataset + ".json")
        with open(file_path, "r", encoding="utf-8") as f:
            article_abstract = []
            data = [x for x in f.readlines()]
            print(len(data))
            with Pool(processes=cpu_core) as p:
                for item in p.imap(parserArticle, data):
                    if len(item["text"]) == 0 or len(item["abstract"]) == 0:
                       continue
                    article_abstract.append(item)
                    for sent in item["text"]:
                        for w in sent:
                            if w == 3:
                                count_unknow_words += 1
                            else:
                                total_words += 1

        save_path = file + "-nums-" + dataset + ".json"
        if os.path.exists(save_path):
            os.remove(save_path)
        
        with open(save_path, "a+", encoding="utf-8") as f:
            for data in article_abstract:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")

        print(file, "length = ", len(article_abstract))
    print("unknow words nums = {} / total_words nums = {}".format(count_unknow_words, total_words))
    print("----"*10)

def convertWSJandCONLL(dataset="wsj"):
    print(dataset, " is converting  words to nums .....")
    root1 = "./datasets"
    for file in ["train", "valid", "test"]:
        file_path = os.path.join(root1 + "/" + dataset, file + "-" + dataset + ".json")
        with open(file_path, "r", encoding="utf-8") as f:
            wsj_dataset = []
            for line in f.readlines():
                line = json.loads(line)
                text = [convert2id(sent) for sent in line["text"]]
                text_pos = [convertPos2id(p) for p in line["pos"]]
                text_head = line["head"]
                try:
                    relations = [convertRelation2id(sent) for sent in line["relation"]]
                except:
                    relations = []

                wsj_dataset.append(
                    {"text": text,
                     "pos": text_pos,
                     "head": text_head,
                     "relations": relations
                     }
                )
        save_path = file + "-nums-" + dataset + ".json"
        if os.path.exists(save_path):
            os.remove(save_path)
        with open(save_path, "a+", encoding="utf-8") as f:
            for data in wsj_dataset:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")


def convertAG(dataset="ag"):
    print(dataset, " is converting  words to nums .....")
    root1 = "./datasets"
    root2 = "."
    for file in ["train", "valid", "test"]:
        file_path = os.path.join(root1 + "/" + dataset, file + "-" + dataset + ".json")
        with open(file_path, "r", encoding="utf-8") as f:
            wsj_dataset = []
            for line in f.readlines():
                line = json.loads(line)
                text = convert2id(line["text"])
                label = conevertClass2id(line["class"])
                wsj_dataset.append(
                    {"text": [text],
                     "class": label
                     }
                )
        save_path = file + "-nums-" + dataset + ".json"
        if os.path.exists(save_path):
            os.remove(save_path)
        with open(save_path, "a+", encoding="utf-8") as f:
            for data in wsj_dataset:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")


def getunknowwords(dataset):
    count_unknow_words = 0
    total_words = 0
    for file in ["train", "valid", "test"]:
        file_path = file + "-nums-" + dataset + ".json"
        cnt = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                cnt += 1
                text = json.loads(line)
                for sent in text["text"]:
                    for w in sent:
                        if w == 3:
                            count_unknow_words += 1
                        total_words += 1
        print(file, "length = ", cnt)
    print("unknow words nums = {} / total_words nums = {}".format(count_unknow_words, total_words))
    print("----"*10)


if __name__ == '__main__':
    # convert the article 2 nums
    dataset ="cnndm"
    print(dataset)
    if dataset == "cnndm":
        convertTextSummarization(dataset)
    elif dataset == "wsj" or dataset == "conll":
        convertWSJandCONLL(dataset)
        getunknowwords(dataset)
    elif dataset == "ag":
        convertAG("ag")
        getunknowwords(dataset)

    

