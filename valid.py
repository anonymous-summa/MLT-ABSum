import datetime
import os
import time
from torch.autograd import Variable
from train import *
import random
from train import *

class ValidModel():
    def __init__(self):
        self.seq2seqmodel = Seq2SeqModel().to(config.use_gpu)
        self.pt = "cover" if config.is_coverage else "pg"
        self.root_seq2seq_dir = os.path.join("model", config.dataset + "-" + self.pt)

    def _getdata(self):
        path = "../datasets/tmp-16/"+config.dataset
        self.valid_iter = []
        file_path = [x for x in os.listdir(path) if "valid" in x]
        for file_name in file_path:
            file_path = os.path.join(path,file_name)
            print(file_path)
            with open(file_path, "rb") as f:
                self.valid_iter += torch.load(f)

    def _load_model(self, last_time_model):
        state = torch.load(last_time_model, map_location=lambda storage, location: storage)
        self.seq2seqmodel.encoder.load_state_dict(state['encoder_state_dict'])
        self.seq2seqmodel.posTask.load_state_dict(state["pos_state_dict"])
        self.seq2seqmodel.dpTask.load_state_dict(state["dp_state_dict"])
        self.seq2seqmodel.textclassTask.load_state_dict(state['class_state_dict'])
        self.seq2seqmodel.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
        self.seq2seqmodel.reduce_state.load_state_dict(state['reduce_state_dict'])
            

    def valid(self):
        self._getdata()
        model_path = [x for x in os.listdir(self.root_seq2seq_dir)]
        model_path = sorted(model_path)
        index = -60 if not config.is_coverage else 1 
        model_path = model_path[index:]
        # self.valid_iter = self.valid_iter
        print("[INFO] {} | valid batch len = {}".format(datetime.datetime.now(), len(self.valid_iter)))
        print("[INFO] {} | model have {} models".format(datetime.datetime.now(), len(model_path)))
        print ("----" * 10)
        
        best_model_name_loss = {}
        for cnt, mpt in enumerate(model_path):
            last_time_model = os.path.join(self.root_seq2seq_dir, mpt)
            self._load_model(last_time_model)
            self.seq2seqmodel.eval()
            running_avg_loss = 0.0
            with torch.no_grad():
                for idx, batch in enumerate(self.valid_iter):
                    loss = self.seq2seqmodel(batch, "valid")
                    running_avg_loss += loss.item()

            running_avg_loss /= len(self.valid_iter)
            best_model_name_loss[mpt] = running_avg_loss

            print("[INFO] {} | model name = {} | idx = {}/{} | valid avg loss : {}".format(datetime.datetime.now(),last_time_model, cnt, len(model_path), running_avg_loss))

            print ("*******" * 10)
        
        best_model_name_loss = sorted(best_model_name_loss.items(), key=lambda x: x[1])
        for item in best_model_name_loss:
            print ("model_name: {} | loss = {}".format(item[0], item[1]))
        print ("----" * 10)
        best = [x[0] for x in best_model_name_loss]
        print (best)
        print("----"*10)
        print("[INFO] {} | valid have finished !".format(datetime.datetime.now()))



if __name__ == '__main__':
    config.dataset = "cnndm"
    config.max_dec_steps = 100 if config.dataset == "cnndm" else 290
    config.use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.pointer_gen = True
    config.is_coverage = True
    config.batch_size = 16
    valid = ValidModel()
    valid.valid()
