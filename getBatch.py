# _*_coding: utf-8_*_

import json
import os
import sys

import torch

from torch.utils.data import Dataset, DataLoader
import config as config
# sys.maxsize

class MyDataSet(Dataset):
    def __init__(self, path, max_line=sys.maxsize):
        super(MyDataSet, self).__init__()
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                self.data.append(json.loads(line))
                if idx >= max_line:
                    break

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    max_article_length = config.max_enc_steps
    article = {}
    for idx, item in enumerate(batch):  # [batch, ]
        xx, xx_lenth, enc_padding_mask, enc_extend_words = [], [], [], []
        oovs = []
        try:
            for sent, ext in zip(item["text"], item["enc_extend_words"]):
                if len(xx) == max_article_length:
                    break
                total_word = 0
                for i in range(len(sent)):
                    if len(xx) == max_article_length:
                        break
                    xx.append(sent[i])
                    enc_extend_words.append(ext[i])
                    enc_padding_mask.append(1)
                    total_word += 1
                    if ext[i] >= len(config.word2id) and ext[i] not in oovs:
                        oovs.append(ext[i])
                xx_lenth.append(total_word)
        except:
            for sent in item["text"]:
                if len(xx) == max_article_length:
                    break
                total_word = 0
                for w in sent:
                    if len(xx) == max_article_length:
                        break
                    xx.append(w)
                    total_word += 1
                    enc_padding_mask.append(1)
                xx_lenth.append(total_word)

        try:
            dec_abs, dec_padding_mask = [config.word2id["BOS"]], [1]
            dec_extend_words = [config.word2id["BOS"]]
            for sent, dxt in zip(item["abstract"], item["doc_extend_words"]):
                for i in range(len(sent)):
                    dec_abs.append(sent[i])
                    dec_padding_mask.append(1)
                    dec_extend_words.append(dxt[i] if dxt[i] < len(config.word2id)+len(oovs) else config.word2id["UNK"])

            if len(dec_abs) >= config.max_dec_steps:
                dec_abs = dec_abs[:config.max_dec_steps]
                dec_padding_mask = dec_padding_mask[:config.max_dec_steps]
                dec_extend_words = dec_extend_words[:config.max_dec_steps]
            else:
                dec_abs.append(config.word2id["EOS"])
                dec_padding_mask.append(1)
                dec_extend_words.append(config.word2id["EOS"])
        except:
            dec_abs = []
            dec_padding_mask = []
            dec_extend_words = []

        try:
            xx_pos = [w for sent in item["pos"] for w in sent]
        except:
            xx_pos = []
        try:
            xx_dp = ([w for sent in item["head"] for w in sent], [w for sent in item["relations"] for w in sent])
        except:
            xx_dp = []

        try:
            xx_label = [item["class"]]
        except:
            xx_label = []
        
        article[idx] = (xx_lenth, xx, dec_abs, xx_pos, xx_dp, xx_label, enc_padding_mask, dec_padding_mask, enc_extend_words, dec_extend_words, item["oovs"][:len(oovs)] if "oovs" in item.keys() else [])
    article = sorted(article.items(), key=lambda x: sum(x[1][0]), reverse=True)

    total_length, x, y, x_pos, x_dp, x_label, enc_padding_masks, dec_padding_masks = [], [], [], [], [], [], [], []
    enc_extend_words, dec_extend_words = [], []
    oovs = []

    for key, v in article:
        total_length.append(v[0])  # article length
        x.append(torch.tensor(v[1]))  # article
        y.append(torch.tensor(v[2]))  # article abstract
        x_pos.append(torch.tensor(v[3]))  # article pos
        x_dp.append(v[4])
        x_label.append(torch.tensor(v[5]))
        enc_padding_masks.append(torch.tensor(v[6]))
        dec_padding_masks.append(torch.tensor(v[7]))
        enc_extend_words.append(torch.tensor(v[8]))
        dec_extend_words.append(torch.tensor(v[9]))
        oovs.append(v[10])

    
    batch  = {
        "article": x, 
        "abstract": y, 
        "length": total_length, 
        "article_len": [sum(x) for x in total_length],
        "pos": x_pos, 
        "head": x_dp, 
        "class": x_label, 
        "enc_padding_mask": enc_padding_masks, 
        "dec_padding_mask": dec_padding_masks,
        "enc_extend_words": enc_extend_words,
        "dec_extend_words":dec_extend_words,
        "oovs": oovs,
        "oovs_len":max([len(x) for x in oovs]),
        }
    return batch

def getBatch(op="train"):
    root = "../datasets"
    if op == "train":
        data_iter = DataLoader(dataset=MyDataSet(os.path.join(root, op + "-nums-" + config.dataset + ".json")), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        data_iter = DataLoader(dataset=MyDataSet(os.path.join(root, op + "-nums-" + config.dataset + ".json")), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    return data_iter

