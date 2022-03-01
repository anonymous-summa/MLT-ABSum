# _*_coding: utf-8_*_

import datetime
import os

from torch.nn.utils import clip_grad_norm_

from classTask import TextClassTask
from dpTask import DPTask, Variable
from posTask import PosTask
from encoder import *
from decoder import *
from getBatch import *
import random
import time

from torch.utils.data import Dataset, DataLoader

class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder().eval()
        self.posTask = PosTask().eval()
        self.dpTask = DPTask().eval()
        self.textclassTask = TextClassTask().eval()

        self.reduce_state = ReduceState()
        self.decoder = Decoder()
        self.decoder.embedding.weight = self.reduce_state.embedding.weight
        self.teacher_forcing_ratio = config.teacher_force

    def forward(self, batch, op="train"):
        # input text
        article_len = batch["article_len"]
        inputs_text = pad_sequence(batch["article"], batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)
        
        with torch.no_grad():
            pos_outputs, _ = self.encoder(inputs_text, article_len)
            _, pos_score = self.posTask(pos_outputs)
            pos_results = self.posTask.predict(pos_score.view(pos_outputs.shape[0], pos_outputs.shape[1], -1))

            dp_outputs, dp_hidden = self.dpTask(inputs_text, pos_outputs, pos_results)
            dp_heads = batch["heads"]
            dp_relations = batch["relations"]

            (class_outputs, class_hidden), class_results = self.textclassTask(inputs_text, dp_outputs, dp_heads, dp_relations, pos_results)
        
        
        oovs_len = batch["oovs_len"]
        enc_padding_mask = pad_sequence(batch["enc_padding_mask"], padding_value=0, batch_first=True).to(config.use_gpu)
        dec_padding_mask = pad_sequence(batch["dec_padding_mask"], padding_value=0, batch_first=True).to(config.use_gpu)
        target_batch = pad_sequence(batch["abstract"], padding_value=config.word2id["PAD"], batch_first=True).to(config.use_gpu) # have unk

        extra_zeros = torch.zeros(inputs_text.shape[0], oovs_len).to(config.use_gpu) if oovs_len > 0 else None # greater than word2id dict
        enc_batch_extend_vocab = pad_sequence(batch["enc_extend_words"], padding_value=config.word2id["PAD"], batch_first=True).to(config.use_gpu)
        dec_batch_extend_vocab = pad_sequence(batch["dec_extend_words"], padding_value=config.word2id["PAD"], batch_first=True).to(config.use_gpu) # have all

        encoder_outputs, encoder_feature, s_t_1 = self.reduce_state(inputs_text, class_outputs, class_hidden, dp_outputs, dp_heads, dp_relations, pos_outputs, pos_results)
        
        # print(encoder_outputs.shape)

        # finished changed
        c_t_1 = torch.zeros((encoder_outputs.shape[0], 2 * config.hidden_dim)).to(config.use_gpu)
        
        coverage = torch.zeros(encoder_outputs.shape[0], encoder_outputs.shape[1]).to(config.use_gpu) if config.is_coverage else None

        dec_length_var = torch.tensor([len(x) for x in batch["abstract"]]).to(config.use_gpu)

        #  Teacher forcing   
        y_t_1 = target_batch[:, 0]

        step_losses = []

        for di in range(1, min(dec_padding_mask.shape[1], config.max_dec_steps)):
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1, enc_batch_extend_vocab, extra_zeros, coverage, di - 1, op)

            target = dec_batch_extend_vocab[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()

            step_loss = -torch.log(gold_probs + config.eps)  # loss compute

            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)
            
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

            if use_teacher_forcing:
                y_t_1 = target_batch[:, di]
            else:
                predict_words = torch.argmax(final_dist, dim=1)
                y_t_1 = target_batch[:, di]
                y_t_1 = torch.where(predict_words >= len(config.word2id), y_t_1, predict_words)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_length_var
        loss = torch.mean(batch_avg_loss)
        # teacher Focing 

        return loss


class TrainSeq2Seq(object):
    def __init__(self):
        self.seq2seqmodel = Seq2SeqModel().to(config.use_gpu)
        if not os.path.exists("model"):
            os.mkdir("model")
        self.root_seq2seq_dir = os.path.join("model", config.dataset + "-" + ("cover" if config.is_coverage else "pg"))
        if not os.path.exists(self.root_seq2seq_dir):
            os.mkdir(self.root_seq2seq_dir)
        self.root_class_dir = os.path.join("model", "class_model")

    def _getdata(self):
        path = "../datasets/tmp-"+str(config.batch_size)+"/"+config.dataset
        self.train_iter = []
        file_path = [x for x in os.listdir(path) if "train" in x]
        for file_name in file_path:
            file_path = os.path.join(path,file_name)
            print(file_path)
            with open(file_path, "rb") as f:
                self.train_iter += torch.load(f)

    def _save_model(self, iter2, running_avg_loss):
        state = {
            'iter': iter2, 
            'encoder_state_dict': self.seq2seqmodel.encoder.state_dict(), 
            'pos_state_dict': self.seq2seqmodel.posTask.state_dict(), 
            'dp_state_dict': self.seq2seqmodel.dpTask.state_dict(), 
            'class_state_dict': self.seq2seqmodel.textclassTask.state_dict(), 
            'decoder_state_dict': self.seq2seqmodel.decoder.state_dict(), 
            'reduce_state_dict': self.seq2seqmodel.reduce_state.state_dict(), 
            'optimizer': self.optimizer.state_dict(), 
            'current_loss': running_avg_loss
            }
            
        torch.save(state, self.root_seq2seq_dir + "/" + config.dataset + "-seq2seq_model" + "-" + str(int(time.time())))

    def _load_model(self):
        model_path = os.listdir(self.root_seq2seq_dir)
        if len(model_path) > 0:
            model_path.sort()
            last_time_model = os.path.join(self.root_seq2seq_dir, model_path[-1])
            print(last_time_model)
            print("----" * 20)
            print("loading seq2seq model parameters ")
            state = torch.load(last_time_model, map_location=lambda storage, location: storage)
            self.seq2seqmodel.encoder.load_state_dict(state['encoder_state_dict'])
            self.seq2seqmodel.posTask.load_state_dict(state["pos_state_dict"])
            self.seq2seqmodel.dpTask.load_state_dict(state["dp_state_dict"])
            self.seq2seqmodel.textclassTask.load_state_dict(state['class_state_dict'])
            self.seq2seqmodel.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.seq2seqmodel.reduce_state.load_state_dict(state['reduce_state_dict'])
            if not config.is_coverage:
               self.optimizer.load_state_dict(state["optimizer"])
            return state["iter"] + 1, state["current_loss"]
        else:
            print("loading class model parameters ")
            state = torch.load(self.root_class_dir, map_location=lambda storage, location: storage)
            self.seq2seqmodel.encoder.load_state_dict(state['encoder_state_dict'])
            self.seq2seqmodel.posTask.load_state_dict(state["pos_state_dict"])
            self.seq2seqmodel.dpTask.load_state_dict(state["dp_state_dict"])
            self.seq2seqmodel.textclassTask.load_state_dict(state['class_state_dict'])
            return 0, None

    def _setup_train(self):
        params = list(self.seq2seqmodel.decoder.parameters()) + list(self.seq2seqmodel.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = torch.optim.Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)


    def calc_running_avg_loss(self, running_avg_loss, loss):
        decay = 0.99
        if running_avg_loss is None:
            running_avg_loss = loss
        else:
            running_avg_loss = min(running_avg_loss * decay + (1 - decay) * loss, 12)
        return running_avg_loss

    def train(self):
        self._getdata()
        self._setup_train()
        print("[INFO] {} | train batch len = {}".format(datetime.datetime.now(), len(self.train_iter)))
        iter2, running_avg_loss = self._load_model()
        print(iter2, running_avg_loss)

        print("****" * 20)
        if config.is_coverage:
            config.max_iter = iter2 + 5000

        while iter2 < config.max_iter:
            for idx, batch in enumerate(self.train_iter):
                self.seq2seqmodel.encoder.eval()
                self.seq2seqmodel.posTask.eval()
                self.seq2seqmodel.dpTask.eval()
                self.seq2seqmodel.textclassTask.eval()

                self.seq2seqmodel.reduce_state.train()
                self.seq2seqmodel.decoder.train()

                self.optimizer.zero_grad()
                train_loss = self.seq2seqmodel(batch, "train")
                
                train_loss.backward()

                clip_grad_norm_(self.seq2seqmodel.decoder.parameters(), config.max_grad_norm)
                clip_grad_norm_(self.seq2seqmodel.reduce_state.parameters(), config.max_grad_norm)

                self.optimizer.step()

                running_avg_loss = self.calc_running_avg_loss(running_avg_loss, train_loss.item())
                
                print_loss_time = 500 if not config.is_coverage else 100

                if iter2 % print_loss_time == 0:
                    print("[INFO] {} | Epoch : {} | process:{}/{} | train_loss : {} | running_avg_loss : {}".format(datetime.datetime.now(), iter2, idx, len(self.train_iter), train_loss.item(), running_avg_loss))

                save_model_time = 2500 if not config.is_coverage else 100

                if iter2 % save_model_time == 0:
                    self._save_model(iter2, running_avg_loss)
                    print("best model have saved!")
                    print("**" * 20)
                if iter2 == config.max_iter:
                    break
                iter2 += 1

        print("[INFO] {} | trainging model is finished ".format(datetime.datetime.now()))
        print("**" * 20)

if __name__ == '__main__':
    # config.dataset = "cnndm"
    # config.max_dec_steps = 100 if config.dataset == "cnndm" else 290
    
    # config.use_gpu = "cuda:0" if torch.cuda.is_available() else "cpu"

    # config.pointer_gen = True
    # config.is_coverage = True
    # config.batch_size = 16
    # config.max_iter = 250000
    train = TrainSeq2Seq()
    train.train()
