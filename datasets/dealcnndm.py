# _*_coding: utf-8_*_

import json
import os
from multiprocessing.pool import Pool
import multiprocessing
import spacy

cpu_core = multiprocessing.cpu_count()

nlp = spacy.load('en_core_web_sm')  # 


def parserCutwords(article):
    text = article["text"]
    abstract = article["abstract"]
    # sentence split and tokenization
    doc = nlp(text)   
    text_root = [[str(x) for x in sent] for sent in doc.sents]
    abstr = nlp(abstract)
    text_abstract = [[str(x) for x in sent] for sent in abstr.sents]
    return {
        "text": text_root,
        "abstract": text_abstract,
    }


def articleCutwords():
    for file_pth in ["train", "valid", "test"]:
        article_abstract = []
        artilce_path = os.path.join(path, file_pth + ".txt.src")
        ref_path = os.path.join(path, file_pth + ".txt.tgt.tagged")
        
        with open(artilce_path, "r", encoding="utf-8") as f1, open(ref_path, "r", encoding="utf-8") as f2:
            data = [{"text": text, "abstract": ref} for text, ref in zip(f1.readlines(), f2.readlines())]
            with Pool(processes=cpu_core) as p:
                for item in p.imap(parserCutwords, data):
                    article_abstract.append(
                        item
                    )
        save_path = file_pth + "-cnndm.json"
        with open(save_path, "w+", encoding="utf-8") as f:
            for data in article_abstract:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")

if __name__ == '__main__':
    articleCutwords()
