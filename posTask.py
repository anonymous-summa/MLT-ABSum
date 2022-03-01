# _*_coding: utf-8_*_

import datetime
import os
import random

import torch.nn.functional as F

from sklearn.metrics import *

from getBatch import getBatch

from encoder import *

random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)


def getAccPos(pred, y):
    pred = pred.tolist()
    total_p = 0.0
    total_f1 = 0.0
    for predict, item in zip(pred, y):
        item_len = len(item)
        total_p += precision_score(item, predict[:item_len], average="micro")
        try:
            total_f1 += f1_score(item, predict[:item_len], average="micro")
        except:
            pass

    return total_p / len(y), total_f1 / len(y)


class PosTask(nn.Module):
    def __init__(self):
        super(PosTask, self).__init__()
        self.pos_class = len(config.pos2id)
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.fc = nn.Linear(config.hidden_dim * 2, self.pos_class)

    def forward(self, encoder_outputs):
        inputs_shape = encoder_outputs.shape
        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        pos_feature = self.W_h(encoder_feature)
        outputs = self.fc(encoder_feature)
        tag_score = F.log_softmax(outputs, dim=1)
        pos_hidden = pos_feature.view(inputs_shape)  # pos features shape =[ batch,seq_len,hidden*2]
        return pos_hidden, tag_score

    def predict(self, outputs):
        return torch.argmax(outputs, dim=2)


class TrainTestPos():
    def __init__(self):
        self.encoder = Encoder().to(config.use_gpu)
        self.posTask = PosTask().to(config.use_gpu)
        if not os.path.exists("model"):
            os.mkdir("model")
        self.root_dir = os.path.join("model", "pos_model")

    def _getdata(self, op="train"):
        if op == "train":
            self.train_iter = getBatch(op="train")
            self.valid_iter = getBatch(op="valid")
        else:
            self.test_iter = getBatch(op="test")

    def _save_model(self, acc, iter):
        state = {'iter': iter, 'encoder_state_dict': self.encoder.state_dict(), 'pos_state_dict': self.posTask.state_dict(), 'optimizer': self.optimizer.state_dict(), 'pos_acc': acc}
        torch.save(state, self.root_dir)

    def _load_Model(self, op="train"):
        try:
            print("loading pos model parameters")
            state = torch.load(self.root_dir, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.posTask.load_state_dict(state["pos_state_dict"])
            if op == "train":
                self.optimizer.load_state_dict(state["optimizer"])
            return state["iter"], state["pos_acc"]
        except:
            return 0, 0.0

    def _setup_train(self):
        self.epochs = 5
        self.lr = 0.001
        params = list(self.encoder.parameters()) + list(self.posTask.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        self.loss = nn.NLLLoss()

    def _train_one_batch(self, batch):
        article_len = [sum(x) for x in batch["length"]]
        inputs_text = pad_sequence(batch["article"], batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)
        encoder_outputs, encoder_hidden = self.encoder(inputs_text, article_len)
        _, outputs = self.posTask(encoder_outputs)
        y_pos = pad_sequence(batch["pos"], batch_first=True, padding_value=config.pos2id["."]).to(config.use_gpu)
        loss_val = self.loss(outputs, y_pos.view(-1))
        predict = self.posTask.predict(outputs.view(encoder_outputs.shape[0], encoder_outputs.shape[1], -1))
        acc, _ = getAccPos(predict, batch["pos"])
        return loss_val, acc

    def evaluate(self):
        self.encoder.eval()
        self.posTask.eval()
        valid_len = len(self.valid_iter)
        valid_total_loss = 0.0
        valid_total_acc = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(self.valid_iter):
                loss_val, acc = self._train_one_batch(batch)
                valid_total_loss += loss_val.item()
                valid_total_acc += acc
        return valid_total_loss / valid_len, valid_total_acc / valid_len

    def train(self):
        self._setup_train()
        self._getdata("train")
        iter, best_valid_acc = self._load_Model("train")
        print("train batch = {}  | valid batch len = {}".format(len(self.train_iter), len(self.valid_iter)))
        for epoch in range(iter, self.epochs):
            train_acc = 0.
            for idx, batch in enumerate(self.train_iter):
                self.encoder.train()
                self.posTask.train()
                self.optimizer.zero_grad()
                train_loss, train_acc = self._train_one_batch(batch)
                train_loss.backward()
                self.optimizer.step()
                if idx % 400 == 0:
                    valid_loss, valid_acc = self.evaluate()
                    print("[INFO] {} | Epoch : {} | process:{}/{} | train_acc : {} | valid_acc : {}".format(datetime.datetime.now(), epoch, idx, len(self.train_iter), train_acc, valid_acc))
                    if best_valid_acc < valid_acc:
                        best_valid_acc = valid_acc
                        self._save_model(best_valid_acc, epoch)
                        print("best model have saved!")
                    print("**" * 20)
            valid_loss, valid_acc = self.evaluate()

            print("[INFO] {} | Epoch : {} |process:{}/{} | train_acc : {} | valid_acc : {}".format(datetime.datetime.now(), epoch, len(self.train_iter), len(self.train_iter), train_acc, valid_acc))
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                self._save_model(best_valid_acc, epoch)
                print("best model have saved!")
            print("***" * 20)

        print("[INFO] {} trainging model is finished ".format(datetime.datetime.now()))
        print("**" * 20)

    def test(self):
        self._load_Model("test")
        self._getdata("test")
        self.encoder.eval()
        self.posTask.eval()
        test_acc, test_f1, total_article = 0.0, 0.0, 0
        print("[INFO] {} | test data batch length = {}".format(datetime.datetime.now(), len(self.test_iter)))
        print("****" * 10)

        with torch.no_grad():
            for idx, batch in enumerate(self.test_iter):
                total_article += len(batch["article"])
                article_len = [sum(x) for x in batch["length"]]
                inputs_text = pad_sequence(batch["article"], batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)
                encoder_outputs, encoder_hidden = self.encoder(inputs_text, article_len)
                _, outputs = self.posTask(encoder_outputs)
                predict = self.posTask.predict(outputs.view(encoder_outputs.shape[0], encoder_outputs.shape[1], -1))
                acc, f1 = getAccPos(predict, batch["pos"])
                test_acc += acc
                test_f1 += f1
            print("[INFO] {} | total article = {} | test acc = {} | test f1 = {}".format(datetime.datetime.now(), total_article, test_acc / len(self.test_iter), test_f1 / len(self.test_iter)))
            print("******" * 10)


if __name__ == '__main__':
    config.dataset = "wsj"
    config.use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    op = "test"
    if op == "train":
        config.batch_size = 16
        train = TrainTestPos()
        train.train()
    else:
        config.batch_size = 1
        test = TrainTestPos()
        test.test()
