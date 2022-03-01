# _*_coding: utf-8_*_

import datetime
import os
import random

from dpTask import DPTask, parse_proj, getAccDp, Parameter
from posTask import PosTask, getAccPos
from getBatch import getBatch

from sklearn.metrics import *

from encoder import *

random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)


def getAcc(pred, y):
    total_p = precision_score(pred, y, average="micro")
    total_f1 = f1_score(pred, y, average="macro")
    return total_p, total_f1


class TextClassTask(nn.Module):
    def __init__(self):
        super(TextClassTask, self).__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        self.pos_embedding = nn.Embedding(len(config.pos2id), config.emb_dim)
        self.dp_results_embedding = nn.Embedding(len(config.relation2id), config.emb_dim)

        self.class_lstm = nn.LSTM(config.hidden_dim*2+config.emb_dim*2, config.hidden_dim, batch_first=True, bidirectional=True)

        self.class_nums = len(config.class2id)
        self.class_fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)

        self.w = Parameter((config.hidden_dim * 2, config.hidden_dim * 2))
        self.c_q = Parameter((config.hidden_dim * 2, 1))

        self.fc = nn.Linear(config.hidden_dim * 2, self.class_nums)
        
    def get_dp_label_embedding(self, batch_heads, batch_relations):
        batch_arc_embeded = []
        for i in range(len(batch_heads)): # the item of batch
            heads = batch_heads[i]
            arcs = batch_relations[i]
            doc_arc_embedd = []
            for head, arc in zip(heads, arcs):
                arc_in = torch.tensor(arc).to(config.use_gpu)
                
                arc_embedded = self.dp_results_embedding(arc_in).cpu() # 20,128
                new_arc_embedded = torch.zeros(arc_embedded.shape).cpu()
                for i, h in enumerate(head):
                    if h > 0:
                        new_arc_embedded[i,:] += 0.5*arc_embedded[i,:]
                        new_arc_embedded[h-1] += 0.5*arc_embedded[i,:]
                    else:
                        new_arc_embedded[i] += arc_embedded[i, :]
                doc_arc_embedd.append(new_arc_embedded)
            doc_arc_embedd = torch.cat(doc_arc_embedd,dim=0)
            batch_arc_embeded.append(doc_arc_embedd)

        batch_arc_embeded = pad_sequence(batch_arc_embeded, batch_first=True, padding_value=0).to(config.use_gpu)
        return batch_arc_embeded

    def forward(self, inputs_text, dp_outputs, dp_heads, dp_relations, pos_results):
        word_embedded = self.embedding(inputs_text)
        arc_embedded = self.get_dp_label_embedding(dp_heads,dp_relations)
        pos_embedded = self.pos_embedding(pos_results)
        # print(word_embedded.shape, pos_embedded.shape, arc_embedded.shape)
        
        class_in = torch.cat((dp_outputs, word_embedded, arc_embedded+pos_embedded),dim=2)
        # print(class_in.shape)
        class_outputs, class_hidden = self.class_lstm(class_in)
        # attention
        """
            u_{it} = tanh(wx+b)
            \alpha_{it} = \frac{exp(u_{it}^T u_{w})}{\sum_{t}exp(u_{it}^T u_{w})}
            s_i = \sigma_t
        """
        u = torch.tanh(torch.matmul(class_outputs, self.w))
        atten_score = torch.softmax(torch.matmul(u, self.c_q), dim=1)
        class_outputs = class_outputs * atten_score
        final_state = torch.sum(class_outputs, dim=1)

        class_results = self.fc(final_state)  # [batch, class_nums]

        return (class_outputs, class_hidden), class_results


class TainTestClass():
    def __init__(self):
        self.encoder = Encoder().to(config.use_gpu)
        self.posTask = PosTask().to(config.use_gpu)
        self.dpTask = DPTask().to(config.use_gpu)
        self.textclassTask = TextClassTask().to(config.use_gpu)

        self.root_dp_dir = os.path.join("model", "dp_model")
        self.root_class_dir = os.path.join("model", "class_model")
    
    def load_data(self, op):
        data_iter = []
        path = "tmp-" + str(config.batch_size)
        path = os.path.join(path,config.dataset)
        for file_name in os.listdir(path):
            if op in file_name:
                file_path = os.path.join(path,file_name)
                with open(file_path,"rb") as f:
                    data_iter += torch.load(f)
        return data_iter

    def _getdata(self, op="train"):
        if op == "train":
            self.train_iter = self.load_data("train")
            self.valid_iter = self.load_data("valid")
        else:
            self.test_iter = self.load_data("test")

    def _setup_train(self):
        self.epochs = 5
        self.lr = 0.001
        params = list(self.textclassTask.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        self.loss = nn.CrossEntropyLoss()

    def _save_model(self, class_acc, iter):
        state = {
            'iter': iter, 
            'encoder_state_dict': self.encoder.state_dict(), 
            'pos_state_dict': self.posTask.state_dict(), 
            'dp_state_dict': self.dpTask.state_dict(), 
            'class_state_dict': self.textclassTask.state_dict(), 
            'optimizer': self.optimizer.state_dict(), 
            'class_acc': class_acc}
        torch.save(state, self.root_class_dir)

    def _load_model(self, op="train"):
        if not os.path.exists(self.root_class_dir) and op == "train":
            print("loading dp model parameters")
            state = torch.load(self.root_dp_dir, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.posTask.load_state_dict(state["pos_state_dict"])
            self.dpTask.load_state_dict(state["dp_state_dict"])
            return 0, -1
        else:
            print("loading text class model parameters ")
            state = torch.load(self.root_class_dir, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.posTask.load_state_dict(state["pos_state_dict"])
            self.dpTask.load_state_dict(state["dp_state_dict"])
            self.textclassTask.load_state_dict(state["class_state_dict"])
            if op == "train":
                self.optimizer.load_state_dict(state["optimizer"])
            return state["iter"], state["class_acc"]

    def _train_one_batch(self, batch):
        with torch.no_grad():
            article_len = batch["article_len"]

            inputs_text = pad_sequence(batch["article"], batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)

            encoder_outputs, _ = self.encoder(inputs_text, article_len)
            _, pos_score = self.posTask(encoder_outputs)
            pos_results = self.posTask.predict(pos_score.view(encoder_outputs.shape[0], encoder_outputs.shape[1], -1))

            dp_outputs, dp_hidden = self.dpTask(inputs_text, encoder_outputs, pos_results)
            dp_heads = batch["heads"]
            dp_relations = batch["relations"]

        _, class_results = self.textclassTask(inputs_text, dp_outputs, dp_heads, dp_relations, pos_results)

        y_label = torch.tensor(batch["class"]).to(config.use_gpu)
        class_loss = self.loss(class_results, y_label)
        pre_label = torch.argmax(torch.softmax(class_results, dim=1), dim=1)
        class_acc, class_f1 = getAcc(pre_label.tolist(), batch["class"])
        return class_loss, class_acc, class_f1

    def _evaluate(self):
        self.encoder.eval()
        self.posTask.eval()
        self.dpTask.eval()
        self.textclassTask.eval()
        batch_len = len(self.valid_iter)
        test_acc, test_f1, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for idx, batch in enumerate(self.valid_iter):
                valid_loss, valid_acc, valid_f1 = self._train_one_batch(batch)
                total_loss += valid_loss.item()
                test_acc += valid_acc
                test_f1 += valid_f1
        return total_loss / batch_len, test_acc / batch_len, test_f1 / batch_len

    def train(self):
        self._setup_train()
        self._getdata("train")
        iter2, best_valid_acc = self._load_model("train")
        print("train batch len = {}  | valid batch len = {}".format(len(self.train_iter), len(self.valid_iter)))
        if best_valid_acc == -1:
            iter2, best_valid_acc = 0, 0.0
        for epoch in range(iter2, self.epochs):
            train_acc = 0.0
            for idx, batch in enumerate(self.train_iter):
                self.encoder.eval()
                self.posTask.eval()
                self.dpTask.eval()
                self.textclassTask.train()
                self.optimizer.zero_grad()
                train_loss, train_acc, train_f1 = self._train_one_batch(batch)
                train_loss.backward()
                self.optimizer.step()

                if idx % 1000 == 0:
                    valid_loss, valid_acc, valid_f1 = self._evaluate()
                    print("[INFO] {} | Epoch : {} | process:{}/{} | train_acc : {} | valid_acc : {}".format(datetime.datetime.now(), epoch, idx, len(self.train_iter), train_acc, valid_acc))
                    if best_valid_acc < valid_acc:
                        best_valid_acc = valid_acc
                        self._save_model(best_valid_acc, epoch)
                        print("best model have saved!")
                    print("**" * 20)
            valid_loss, valid_acc, valid_f1 = self._evaluate()

            print("[INFO] {} | Epoch : {} | process:{}/{} | train_acc : {} | valid_acc : {}".format(datetime.datetime.now(), epoch, len(self.train_iter), len(self.train_iter), train_acc, valid_acc))
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                self._save_model(best_valid_acc, epoch)
                print("best model have saved!")
            print("***" * 20)

        print("[INFO] {} trainging model is finished ".format(datetime.datetime.now()))
        print("**" * 20)

    def test(self):
        self._load_model("test")
        self._getdata("test")
        self.encoder.eval()
        self.posTask.eval()
        self.dpTask.eval()
        self.textclassTask.eval()
        print("[INFO] {} | test data batch length = {}".format(datetime.datetime.now(), len(self.test_iter)))
        pred, y_true = [], []
        with torch.no_grad():
            for idx, batch in enumerate(self.test_iter):
                article_len = batch["article_len"]

                inputs_text = pad_sequence(batch["article"], batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)

                encoder_outputs, _ = self.encoder(inputs_text, article_len)
                _, pos_score = self.posTask(encoder_outputs)
                pos_results = self.posTask.predict(pos_score.view(encoder_outputs.shape[0], encoder_outputs.shape[1], -1))

                dp_outputs, dp_hidden = self.dpTask(inputs_text, encoder_outputs, pos_results)
                dp_heads = batch["heads"]
                dp_relations = batch["relations"]

                _, class_results = self.textclassTask(inputs_text, dp_outputs, dp_heads, dp_relations, pos_results)
                pre_label = torch.argmax(torch.softmax(class_results, dim=1), dim=1)
                y_label = torch.tensor(batch["class"])
                for w, y in zip(pre_label.tolist(), y_label.tolist()):
                    pred.append(w)
                    y_true.append(y)
            test_acc, test_f1 = getAcc(pred, y_true)
            print("[INFO] {} |  test acc = {} | test f1 = {}".format(datetime.datetime.now(), test_acc, test_f1))
            print("******" * 10)


def testOtherTask(task="pos"):
    encoder = Encoder().to(config.use_gpu)
    posTask = PosTask().to(config.use_gpu)
    dpTask = DPTask().to(config.use_gpu)
    
    root_class_dir = os.path.join("model", "class_model")

    state = torch.load(root_class_dir, map_location=lambda storage, location: storage)
    encoder.load_state_dict(state['encoder_state_dict'])
    posTask.load_state_dict(state["pos_state_dict"])
    dpTask.load_state_dict(state["dp_state_dict"])

    encoder.eval()
    posTask.eval()
    dpTask.eval()

    if task == "pos":
        config.dataset = "wsj"
        config.batch_size = 1
        test_iter = getBatch(op="test")
        test_acc, test_f1, total_article = 0.0, 0.0, 0
        print("[INFO] {} | test data batch length = {}".format(datetime.datetime.now(), len(test_iter)), "\n", "****" * 10)

        with torch.no_grad():
            for idx, batch in enumerate(test_iter):
                total_article += len(batch["article"])
                article_len = [sum(x) for x in batch["length"]]
                inputs_text = pad_sequence(batch["article"], batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)
                encoder_outputs, encoder_hidden = encoder(inputs_text, article_len)
                _, outputs = posTask(encoder_outputs)
                predict = posTask.predict(outputs.view(encoder_outputs.shape[0], encoder_outputs.shape[1], -1))
                acc, f1 = getAccPos(predict, batch["pos"])
                test_acc += acc
                test_f1 += f1
            print("[INFO] {} | total article = {} | test acc = {} | test f1 = {}".format(datetime.datetime.now(), total_article, test_acc / len(test_iter), test_f1 / len(test_iter)))
            print("******" * 10)

    if task == "dp":
        config.dataset = "conll"
        config.batch_size = 1
        test_iter = getBatch(op="test")
        print("[INFO] {} | test data batch length = {}".format(datetime.datetime.now(), len(test_iter)))
        print("****" * 10)
        uas_acc, las_acc = 0.0, 0.0 
        total_article = 0 

        with torch.no_grad():
            for idx, batch in enumerate(test_iter):
                article_len = [sum(x) for x in batch["length"]]
                inputs_text = pad_sequence(batch["article"], batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)
                encoder_outputs, encoder_hidden = encoder(inputs_text, article_len)
                _, pos_tag_score = posTask(encoder_outputs)
                pos_results = posTask.predict(pos_tag_score.view(encoder_outputs.shape[0], encoder_outputs.shape[1], -1))
                dp_outputs, hidden = dpTask(inputs_text, encoder_outputs, pos_results)

                for i in range(dp_outputs.shape[0]):
                    total_article += 1
                    seq_len = len(batch["article"][i])
                    dp_in = dp_outputs[i, :seq_len, :] # not contain viture root
                    scores, exprs,dp_in = dpTask.getscore(dp_in)
                    heads = parse_proj(scores)
                    rels = []
                    for idx, h in enumerate(heads):
                        if idx == 0: continue
                        rexprs = dpTask.getlabelscore(dp_in, h, idx)
                        relation = torch.argmax(rexprs)
                        rels.append(relation.item())
                    gold = batch["head"][i][0]
                    gold_rels = batch["head"][i][1]
                    correct_head = [1 if x == y else 0 for x ,y in zip(heads[1:],gold) ]
                    correct_arc = [1 if x == y and rx == ry else 0 for x,y,rx,ry in zip(heads[1:],gold,rels,gold_rels)]
                    uas_acc += sum(correct_head)/len(correct_head)
                    las_acc += sum(correct_arc)/len(correct_arc)

        print("[INFO] {} | total article = {} | test uas = {} | test las = {}".format(datetime.datetime.now(), total_article, uas_acc / total_article, las_acc/total_article))
        print("******" * 10)


if __name__ == '__main__':
    config.dataset = "ag"
    config.use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    op = "test"
    if op == "train":
        config.batch_size = 16
        train = TainTestClass()
        train.train()
    else:
        config.batch_size = 1
        test = TainTestClass()
        test.test()
        testOtherTask("pos")
        testOtherTask("dp")
