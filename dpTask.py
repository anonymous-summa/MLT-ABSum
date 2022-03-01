# _*_coding: utf-8_*_

import datetime
import os
import random

from torch.nn.utils.rnn import pad_sequence

from torch.nn.init import xavier_uniform_

from encoder import *
from posTask import PosTask, getAccPos
from getBatch import getBatch
import config as config
import torch.nn as nn
import torch
import numpy as np

random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)


def parse_proj(scores, gold=None):
    '''
    Parse using Eisner's algorithm.
    '''
    nr, nc = np.shape(scores)
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = nr - 1  # Number of words (excluding root).

    # Initialize CKY table.
    complete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
    incomplete = np.zeros([N + 1, N + 1, 2])  # s, t, direction (right=1).
    # s, t, direction (right=1).
    complete_backtrack = -np.ones([N + 1, N + 1, 2], dtype=int)
    # s, t, direction (right=1).
    incomplete_backtrack = -np.ones([N + 1, N + 1, 2], dtype=int)

    incomplete[0, :, 0] -= np.inf

    # Loop from smaller items to larger items.
    for k in range(1, N + 1):
        for s in range(N - k + 1):
            t = s + k

            # First, create incomplete items.
            # left tree
            incomplete_vals0 = complete[s, s:t, 1] + complete[(s + 1):(t + 1), t, 0] + scores[t, s] + (0.0 if gold is not None and gold[s] == t else 1.0)
            incomplete[s, t, 0] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
            # right tree
            incomplete_vals1 = complete[s, s:t, 1] + complete[(s + 1):(t + 1), t, 0] + scores[s, t] + (0.0 if gold is not None and gold[t] == s else 1.0)
            incomplete[s, t, 1] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            # left tree
            complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            complete[s, t, 0] = np.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            # right tree
            complete_vals1 = incomplete[s, (s + 1):(t + 1), 1] + \
                complete[(s + 1):(t + 1), t, 1]
            complete[s, t, 1] = np.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

    value = complete[0][N][1]
    heads = [-1 for _ in range(N + 1)]  # -np.ones(N+1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    value_proj = 0.0
    for m in range(1, N + 1):
        h = heads[m]
        value_proj += scores[h, m]

    return heads


def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    '''
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
    head of each word.
    '''
    if s == t:
        return
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == 0:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = incomplete_backtrack[s][t][direction]
        if direction == 0:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return
        else:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return


def getAccDp(pred, y):
    correct_tags = [1 if p == y else 0 for p, y in zip(pred[1:], y[1:])]
    return sum(correct_tags) / len(correct_tags)



class DPTask(nn.Module):
    def __init__(self):
        super(DPTask, self).__init__()
        self.pos_embedding = nn.Embedding(len(config.pos2id), config.emb_dim)  # pos result label embeding
        init_wt_normal(self.pos_embedding.weight)
        
        self.word_embedding = nn.Embedding(config.vocab_size, config.emb_dim)  # text word embeding
        init_wt_normal(self.word_embedding.weight)

        self.dp_lstm = nn.LSTM(config.hidden_dim*2 + config.emb_dim*2, config.hidden_dim, batch_first=True, bidirectional=True)

        self.viture_root = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)

        # head mod layer
        self.hiddenFOH = nn.Linear(config.hidden_dim * 2, config.hidden_dim)  # head
        self.hiddenFOM = nn.Linear(config.hidden_dim * 2, config.hidden_dim)  # mod
        self.hidBias = Parameter(config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, 1)
        
        # arc layer
        # if config.labelflag:
        # self.arc_fc = nn.Linear(config.hidden_dim*4,config.hidden_dim*4)
        self.arc_hidden = nn.Linear(config.hidden_dim*4, len(config.relation2id))

    def forward(self, texts, encoder_outputs, word_pos):
        """
        :param encoder_outputs: [batch,seq_len,hidden*2]
        :param hidden:
        :return:
        """
        word_embedded = self.word_embedding(texts)
        pos_embedded = self.pos_embedding(word_pos)
        dp_inputs = torch.cat((encoder_outputs, word_embedded, pos_embedded),dim=2) 
        dp_outputs, hidden = self.dp_lstm(dp_inputs)
        return dp_outputs, hidden


    def _getExprs(self, head, mod):
        i_head_of_j = self.fc(torch.sigmoid(head + mod + self.hidBias))
        return i_head_of_j

    def getscore(self, dp_outputs):
        viture_root_hidden = torch.relu(self.viture_root(torch.sum(dp_outputs, dim=0).unsqueeze(0)))
        dp_outputs = torch.cat([viture_root_hidden, dp_outputs], dim=0)
        sentence_head = self.hiddenFOH(dp_outputs)
        sentence_mod = self.hiddenFOM(dp_outputs)
        seq_len = dp_outputs.shape[0]
        exprs = [[self._getExprs(sentence_head[i], sentence_mod[j]) for j in range(seq_len)] for i in range(seq_len)]
        scores = np.array([[output.cpu().item() for output in exprsRow] for exprsRow in exprs])
        return scores, exprs, dp_outputs
    
    def getlabelscore(self, dp_outputs, i, j):
        re_in = torch.cat((dp_outputs[i, :],dp_outputs[j, :]),dim=-1)
        re_in = re_in.unsqueeze(0)
        relation_out = self.arc_hidden(re_in)
        return  relation_out.squeeze(0)


class TrainTestDP():
    def __init__(self):
        self.encoder = Encoder().to(config.use_gpu)
        self.posTask = PosTask().to(config.use_gpu)
        self.dpTask = DPTask().to(config.use_gpu)
        self.root_dp_dir = os.path.join("model", "dp_model")
        self.root_pos_dir = os.path.join("model", "pos_model")

    def _setup_train(self):
        self.epochs = 5
        params = list(self.dpTask.parameters())
        self.optimizer = torch.optim.Adam(params)

    def _getdata(self, op="train"):
        if op == "train":
            self.train_iter = getBatch(op="train")
            self.valid_iter = getBatch(op="valid")
        else:
            self.test_iter = getBatch(op="test")

    def _save_model(self, dp_acc, iter):
        state = {
            'iter': iter, 
            'encoder_state_dict': self.encoder.state_dict(), 
            'pos_state_dict': self.posTask.state_dict(), 
            'dp_state_dict': self.dpTask.state_dict(), 
            'optimizer': self.optimizer.state_dict(), 
            'dp_acc': dp_acc
            }
        torch.save(state, self.root_dp_dir)

    def _load_model(self, op="train"):
        if not os.path.exists(self.root_dp_dir) and op == "train":
            print("loading encoder and pos model parameters")
            state = torch.load(self.root_pos_dir, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.posTask.load_state_dict(state["pos_state_dict"])
            return state["iter"], -1
        else:
            print("loading dp model parameters")
            state = torch.load(self.root_dp_dir, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.posTask.load_state_dict(state["pos_state_dict"])
            self.dpTask.load_state_dict(state["dp_state_dict"])
            if op == "train":
                self.optimizer.load_state_dict(state["optimizer"])
            return state["iter"], state["dp_acc"]

    def _train_one_batch(self, batch, op="train"):
        with torch.no_grad():
            article_len = batch["article_len"]
            inputs_text = pad_sequence(batch["article"], batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)
            encoder_outputs, _ = self.encoder(inputs_text, article_len)
            pos_features, outputs = self.posTask(encoder_outputs)
            pos_results = self.posTask.predict(outputs.view(encoder_outputs.shape[0], encoder_outputs.shape[1], -1))

        dp_outputs, hidden = self.dpTask(inputs_text, encoder_outputs, pos_results)

        errs = []
        total_acc = 0.0
        r_acc = 0.0
        batch_size = dp_outputs.shape[0]
        for i in range(batch_size):
            seq_len = len(batch["article"][i])
            dp_in = dp_outputs[i, :seq_len, :]
            scores, exprs, dp_in = self.dpTask.getscore(dp_in)
            # print(batch["head"][i])
            # each setence have a root id, which map -1
            gold = [-1] + batch["head"][i][0]
            heads = parse_proj(scores, gold if op == "train" else None)
            e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
            if e > 0:
                errs += [(exprs[h][i] - exprs[g][i]) for i, (h, g) in enumerate(zip(heads, gold)) if h != g]
            
            gold_relation = [-1] + batch["head"][i][1]
            if config.labelflag:
                for idx, head in enumerate(gold):
                    if idx == 0:
                        continue
                    rexprs = self.dpTask.getlabelscore(dp_in, head, idx)
                    rscores = rexprs.tolist()
                    goldLabelInd = gold_relation[idx]
                    wrongLabelInd = torch.argmax(rexprs)
                    if rscores[goldLabelInd] < rscores[wrongLabelInd] + 1:
                        errs += [(rexprs[wrongLabelInd] - rexprs[goldLabelInd]).unsqueeze(0).unsqueeze(0)]
        return errs

    def evaluate(self):
        self.encoder.eval()
        self.posTask.eval()
        self.dpTask.eval()

        uas_acc, las_acc = 0.0, 0.0
        total_article = 0

        with torch.no_grad():
            for idx, batch in enumerate(self.valid_iter):
                article_len = [sum(x) for x in batch["length"]]
                inputs_text = pad_sequence(batch["article"], batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)
                encoder_outputs, encoder_hidden = self.encoder(inputs_text, article_len)
                _, pos_tag_score = self.posTask(encoder_outputs)
                pos_results = self.posTask.predict(pos_tag_score.view(encoder_outputs.shape[0], encoder_outputs.shape[1], -1))
                dp_outputs, hidden = self.dpTask(inputs_text, encoder_outputs, pos_results)

                for i in range(dp_outputs.shape[0]):
                    total_article += 1
                    seq_len = len(batch["article"][i])
                    dp_in = dp_outputs[i, :seq_len, :] # not contain viture root
                    scores, exprs,dp_in = self.dpTask.getscore(dp_in)
                    heads = parse_proj(scores)
                    rels = []
                    for idx, h in enumerate(heads):
                        if idx == 0: continue
                        rexprs = self.dpTask.getlabelscore(dp_in, h, idx)
                        relation = torch.argmax(rexprs)
                        rels.append(relation.item())
                    gold = batch["head"][i][0]
                    gold_rels = batch["head"][i][1]
                    correct_head = [1 if x == y else 0 for x ,y in zip(heads[1:],gold) ]
                    correct_arc = [1 if x == y and rx == ry else 0 for x,y,rx,ry in zip(heads[1:],gold,rels,gold_rels)]
                    uas_acc += sum(correct_head)/len(correct_head)
                    las_acc += sum(correct_arc)/len(correct_arc)

        return  uas_acc / total_article, las_acc/total_article

    def train(self):
        self._setup_train()
        self._getdata("train")
        iter2, best_valid_acc = self._load_model("train")
        if best_valid_acc == -1:
            iter2, best_valid_acc = 0, 0.0
        print("[INFO] {} | train batch len = {}  | valid batch len = {}".format(datetime.datetime.now(), len(self.train_iter), len(self.valid_iter)))
        print("****" * 20)
        for epoch in range(iter2, self.epochs):
            train_loss = 0.0
            for idx, batch in enumerate(self.train_iter):
                self.encoder.eval()
                self.posTask.eval()
                self.dpTask.train()
                self.optimizer.zero_grad()
                errs = self._train_one_batch(batch)

                if len(errs) > 0:
                    loss = torch.sum(torch.cat(errs))
                    train_loss = loss.item()
                    loss.backward()
                    self.optimizer.step()

                if idx % 500 == 0:
                    valid_acc, valid_r_acc = self.evaluate()
                    print("[INFO] {} | Epoch : {} | process: {}/{} | training loss = {} | valid uas: {} | las : {}".format(datetime.datetime.now(), epoch, idx, len(self.train_iter), train_loss, valid_acc, valid_r_acc))
                    if best_valid_acc < valid_acc+valid_r_acc:
                        best_valid_acc = valid_acc+valid_r_acc
                        self._save_model(best_valid_acc, epoch)
                        print("best model have saved!")
                    print("**" * 20)

            valid_acc, valid_r_acc = self.evaluate()

            print("[INFO] {} | Epoch : {} | process:{}/{} | training_loss : {} |  valid uas: {} | las : {}".format(datetime.datetime.now(), epoch, len(self.train_iter), len(self.train_iter), train_loss, valid_acc, valid_r_acc))

            if best_valid_acc < valid_acc+valid_r_acc:
                best_valid_acc = valid_acc+valid_r_acc
                self._save_model(best_valid_acc, epoch)
                print("best model have saved!")
            print("***" * 20)

        print("[INFO] {} trainging model is finished ".format(datetime.datetime.now()))
        print("**" * 20)

    def test(self):
        self._load_model("test")
        self._getdata("test")
        self.encoder.eval()
        self.dpTask.eval()
        self.posTask.eval()
        uas_acc, las_acc = 0.0, 0.0
        print("[INFO] {} | test data batch length = {}".format(datetime.datetime.now(), len(self.test_iter)))
        print("****" * 10)
        with torch.no_grad():
            for idx, batch in enumerate(self.test_iter):
                if idx % 100 == 0:
                    print("[INFO] {} | process = {} / {}".format(datetime.datetime.now(), idx, len(self.test_iter)))
                article_len = [sum(x) for x in batch["length"]]
                inputs_text = pad_sequence(batch["article"], batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)
                encoder_outputs, encoder_hidden = self.encoder(inputs_text, article_len)
                _, pos_tag_score = self.posTask(encoder_outputs)
                pos_results = self.posTask.predict(pos_tag_score.view(encoder_outputs.shape[0], encoder_outputs.shape[1], -1))
                dp_outputs, hidden = self.dpTask(inputs_text, encoder_outputs, pos_results)

                batch_size = dp_outputs.shape[0]
                for i in range(batch_size):
                    seq_len = len(batch["article"][i])
                    dp_in = dp_outputs[i, :seq_len, :] # not contain viture root
    
                    scores, exprs,dp_in = self.dpTask.getscore(dp_in)
                    heads = parse_proj(scores)
                    rels = []
                    for idx, h in enumerate(heads):
                        if idx == 0: continue
                        rexprs = self.dpTask.getlabelscore(dp_in, h, idx)
                        relation = torch.argmax(rexprs)
                        rels.append(relation.item())
                    
                    gold = batch["head"][i][0]
                    gold_rels = batch["head"][i][1]
                    correct_head = [1 if x == y else 0 for x ,y in zip(heads[1:],gold) ]
                    correct_arc = [1 if x == y and rx == ry else 0 for x,y,rx,ry in zip(heads[1:],gold,rels,gold_rels)]
                    uas_acc += sum(correct_head)/len(correct_head)
                    las_acc += sum(correct_arc)/len(correct_arc)
                    
            print("[INFO] {} | total article = {} | uas_acc = {} | las_acc = {}".format(datetime.datetime.now(), len(self.test_iter), uas_acc / len(self.test_iter), las_acc/len(self.test_iter)))
            print("******" * 10)

if __name__ == '__main__':
    config.dataset = "conll"
    config.use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.labelflag = True
    op = "test"
    if op == "train":
        config.batch_size = 16
        dptask = TrainTestDP()
        dptask.train()
    else:
        config.batch_size = 1
        dptask = TrainTestDP()
        dptask.test()
