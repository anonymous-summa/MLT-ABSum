# _*_coding: utf-8_*_


import os
import torch
import json

root_dir = os.path.expanduser("")

# Hyperparameters
hidden_dim = 256
emb_dim = 128
vocab_size = 50000

dataset = "cnndm"

if dataset == "cnndm":
    batch_size = 16
    max_enc_steps = 400
    max_dec_steps = 100
    beam_size = 4
    min_dec_steps = 35
    max_iter = 250000
    
pointer_gen = True
is_coverage = False

lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0
cov_loss_wt = 1.0

eps = 1e-12

use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"

lr_coverage = 0.15

teacher_force = 0.5

word2id = json.load(open(os.path.join("../datasets", "word2id.json"), "r"))
id2word = json.load(open(os.path.join("../datasets", "id2word.json"), "r"))

pos2id = {'#': 0, '$': 1, "''": 2, '(': 3, ')': 4, ',': 5, '-LRB-': 6, '-RRB-': 7, '.': 8, ':': 9, 'AFX': 10, 'CC': 11, 'CD': 12, 'DT': 13, 'EX': 14, 'FW': 15, 'HYPH': 16, 'IN': 17, 'JJ': 18, 'JJR': 19, 'JJS': 20, 'LS': 21, 'MD': 22, 'NFP': 23, 'NN': 24, 'NNP': 25, 'NNPS': 26, 'NNS': 27, 'PDT': 28, 'POS': 29, 'PRP': 30, 'PRP$': 31, 'RB': 32, 'RBR': 33, 'RBS': 34, 'RP': 35, 'SYM': 36, 'TO': 37, 'UH': 38, 'VB': 39, 'VBD': 40, 'VBG': 41, 'VBN': 42, 'VBP': 43, 'VBZ': 44, 'WDT': 45, 'WP': 46, 'WP$': 47, 'WRB': 48, '``': 49}

class2id = {"world": 0, "sports": 1, "business": 2, "sci/tech": 3}

labelflag = True
relation2id = {'neg': 0, 'advcl': 1, 'prt': 2, 'mwe': 3, 'xcomp': 4, 'csubjpass': 5, 'mark': 6, 'rel': 7, 'dobj': 8, 'nsubjpass': 9, 'ROOT': 10, 'adpobj': 11, 'rcmod': 12, 'adp': 13, 'nsubj': 14, 'acomp': 15, 'infmod': 16, 'conj': 17, 'amod': 18, 'iobj': 19, 'num': 20, 'cc': 21, 'compmod': 22, 'auxpass': 23, 'partmod': 24, 'dep': 25, 'p': 26, 'ccomp': 27, 'aux': 28, 'adpmod': 29, 'advmod': 30, 'parataxis': 31, 'nmod': 32, 'attr': 33, 'poss': 34, 'csubj': 35, 'cop': 36, 'det': 37, 'adpcomp': 38, 'expl': 39, 'appos': 40
}