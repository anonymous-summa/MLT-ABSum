# _*_coding: utf-8_*_

import datetime
import os

from multiprocessing.pool import Pool

from dpTask import parse_proj
from getBatch import getBatch
import multiprocessing
from dpTask import DPTask
from posTask import PosTask
from encoder import *
import argparse
import random

random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

# cpu_core = multiprocessing.cpu_count()

class InferenceModel(nn.Module):
    def __init__(self):
        super(InferenceModel, self).__init__()
        self.encoder = Encoder()
        self.posTask = PosTask()
        self.dpTask = DPTask()

    def forward(self, inputs_text, article_len):
        inputs_text = pad_sequence(inputs_text, batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)
        # print(inputs_text)
        encoder_outputs, encoder_hidden = self.encoder(inputs_text, article_len)
        _, pos_tag_score = self.posTask(encoder_outputs)
        pos_results = self.posTask.predict(pos_tag_score.view(encoder_outputs.shape[0], encoder_outputs.shape[1], -1))
        dp_outputs, hidden = self.dpTask(inputs_text, encoder_outputs, pos_results)
        return dp_outputs

def sava_new_batch(data, op):
    path = "../datasets/tmp-" + str(config.batch_size)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path,config.dataset)
    if not os.path.exists(path):
        os.mkdir(path)
    # print(data)
    save_path = os.path.join(path, op + "-nums-" + config.dataset + ".pkl")
    with open(save_path, "wb") as f:
        torch.save(data,f)

def getnerate_batch(data_iter, op, args):

    inferHeadModel = InferenceModel().to(config.use_gpu)
    root_dp_dir = os.path.join("model", "dp_model")
    state = torch.load(root_dp_dir, map_location=lambda storage, location: storage)
    inferHeadModel.encoder.load_state_dict(state['encoder_state_dict'])
    inferHeadModel.posTask.load_state_dict(state["pos_state_dict"])
    inferHeadModel.dpTask.load_state_dict(state["dp_state_dict"])

    inferHeadModel.eval()
    
    print("[INFO] {} | data length = {}".format(datetime.datetime.now(),len(data_iter)))
    print("****"*10)
    args.right = min(len(data_iter), args.right)
    print("[INFO] {} | [left,right) = [{},{})  ".format(datetime.datetime.now(),args.left,args.right))
    print("----"*10)
    with torch.no_grad():
        all_batch = []
        for index, batch in enumerate(data_iter):
            if index>= args.left and index < args.right:
                if index % 100 == 0:
                    print("[INFO] {} | index = {}".format(datetime.datetime.now(), index)) 
                dp_outputs = inferHeadModel(batch["article"], batch["article_len"])
                batch_heads, batch_relations = [], []
                for i in range(dp_outputs.shape[0]): # batch
                    st = 0
                    tmp_heads,tmp_arcs = [], []
                    for l_sent in batch["length"][i]: # batch i length 
                        dp_in = dp_outputs[i, st:st+l_sent, :]
                        st += l_sent
                        scores, exprs, dp_in = inferHeadModel.dpTask.getscore(dp_in) # contain viture root
                        heads = parse_proj(scores, None)
                        rels = []
                        for idx, h in enumerate(heads):
                            if idx == 0: continue
                            rexprs = inferHeadModel.dpTask.getlabelscore(dp_in, h, idx)
                            relation = torch.argmax(rexprs)
                            rels.append(relation.item())
                        tmp_heads.append(heads[1:])
                        tmp_arcs.append(rels)
                        assert len(heads[1:]) == len(rels)
                    batch_heads.append(tmp_heads)
                    batch_relations.append(tmp_arcs)
                batch["heads"] = batch_heads
                batch["relations"] = batch_relations
                all_batch.append(batch)
        sava_new_batch(all_batch,op+"-"+str(args.right))
        all_batch = []
    print("[INFO] {} | [left,right) = [{},{})  ".format(datetime.datetime.now(),args.left,args.right))
    print("****"*10)
    print("inference have finished!")
    print("----"*10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input process [left, right]")
    parser.add_argument("--left", type=int,default=0)
    parser.add_argument("--right", type=int, default=10)
    parser.add_argument("--cuda", default="cpu")
    parser.add_argument("--op", default="test")
    args = parser.parse_args()
    
    config.use_gpu = args.cuda
    config.batch_size = 1 if args.op== "test" else config.batch_size
    data_iter = getBatch(args.op)
    getnerate_batch(data_iter,args.op,args)
