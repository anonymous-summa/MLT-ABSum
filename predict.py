# _*_coding: utf-8_*_

import datetime
import json
import os
import sys

from train import *
import argparse


def make_html_safe(s):
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s


def write_for_rouge(reference_sents, decoded_words, ex_index, _rouge_ref_dir, _rouge_dec_dir):
    decoded_sents = []
    tmp_pre = ""
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError:
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx + 1]
        decoded_words = decoded_words[fst_period_idx + 1:]
        decoded_sents.append(" ".join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
        for idx, sent in enumerate(reference_sents):
            f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

def convert2word(id_list, article_oovs):
    words = []
    for i in id_list:
        try:
            w = config.id2word[str(i)]
        except:
            assert article_oovs is not None
            article_oovs_idx  = i-len(config.word2id)
            try:
                w = article_oovs[article_oovs_idx]
            except:
                print("oovs failed !! ")
        if w != "UNK":
            words.append(w)

    return words

def getrefresults():
    path = os.path.join(os.path.join("../datasets/" + config.dataset, "test-"+config.dataset+".json"))
    reference = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            text = json.loads(line)
            text = text["abstract"]
            text = [" ".join(x).strip() for x in text]
            text = [x for x in text if len(x) > 0]
            reference.append({"abstract":text})
    return reference

class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage
    
    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)
    
    @property
    def latest_token(self):
        return self.tokens[-1]
    
    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class TestSeq2Seq():
    def __init__(self):
        self.seq2seqmodel = Seq2SeqModel().to(config.use_gpu)
        self.reference = getrefresults()

    def _makepath(self, model_name):
        self.results_path = "./results"
        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)

        pt = "cover" if config.is_coverage else "pg"

        self.results_path = os.path.join(self.results_path, config.dataset+"-"+pt+"-"+model_name.split("/")[-1].split("-")[-1])

        if not os.path.exists(self.results_path):
            os.mkdir(self.results_path)
        
        self.predict_path = os.path.join(self.results_path, "predict")
        if not os.path.exists(self.predict_path):
            os.mkdir(self.predict_path)
        
        self.ref_path = os.path.join(self.results_path, "ref")
        if not os.path.exists(self.ref_path):
            os.mkdir(self.ref_path)

    def _getdata(self):
        path = "../datasets/tmp-1/"+config.dataset
        self.test_iter = []
        for file_name in os.listdir(path):
            file_path = os.path.join(path,file_name)
            print(file_path)
            with open(file_path, "rb") as f:
                self.test_iter += torch.load(f)
    
    def _load_model(self, model_name):
        print(model_name)
        print("---"*10)
        state = torch.load(model_name, map_location=lambda storage, location: storage)
        self.seq2seqmodel.encoder.load_state_dict(state['encoder_state_dict'])
        self.seq2seqmodel.posTask.load_state_dict(state["pos_state_dict"])
        self.seq2seqmodel.dpTask.load_state_dict(state["dp_state_dict"])
        self.seq2seqmodel.textclassTask.load_state_dict(state['class_state_dict'])
        self.seq2seqmodel.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
        self.seq2seqmodel.reduce_state.load_state_dict(state['reduce_state_dict'])
    
    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def copy_batch_to_beam_size(self, batch):
        for i in range(1, config.beam_size):
            for k in batch.keys():
                if not isinstance(batch[k],int):
                    batch[k].append(batch[k][0])
                    
        return batch
    
    def beam_search(self, batch, op="test"):
        # copy to beamsize
        # print(batch["article"][0])

        batch = self.copy_batch_to_beam_size(batch)

        article_len = batch["article_len"]
        if config.is_coverage:
            config.min_dec_steps = article_len[0]*0.18
        # print(config.min_dec_steps)
        inputs_text = pad_sequence(batch["article"], batch_first=True, padding_value=config.word2id["PAD"]).to(config.use_gpu)

        pos_outputs, _ = self.seq2seqmodel.encoder(inputs_text, article_len)
        _, pos_score = self.seq2seqmodel.posTask(pos_outputs)
        pos_results = self.seq2seqmodel.posTask.predict(pos_score.view(pos_outputs.shape[0], pos_outputs.shape[1], -1))

        dp_outputs, dp_hidden = self.seq2seqmodel.dpTask(inputs_text, pos_outputs, pos_results)
        dp_heads = batch["heads"]
        dp_relations = batch["relations"]

        (class_outputs, class_hidden), class_results = self.seq2seqmodel.textclassTask(inputs_text, dp_outputs, dp_heads, dp_relations, pos_results)
        
        oovs_len = batch["oovs_len"]
        enc_padding_mask = pad_sequence(batch["enc_padding_mask"], padding_value=0, batch_first=True).to(config.use_gpu)
        

        extra_zeros = torch.zeros(inputs_text.shape[0], oovs_len).to(config.use_gpu) if oovs_len > 0 else None # greater than word2id dict
        enc_batch_extend_vocab = pad_sequence(batch["enc_extend_words"], padding_value=config.word2id["PAD"], batch_first=True).to(config.use_gpu)
        dec_batch_extend_vocab = pad_sequence(batch["dec_extend_words"], padding_value=config.word2id["PAD"], batch_first=True).to(config.use_gpu) # have all
        

        encoder_outputs, encoder_feature, s_t_1 = self.seq2seqmodel.reduce_state(inputs_text, class_outputs, class_hidden, dp_outputs, dp_heads, dp_relations, pos_outputs, pos_results)

        dec_h, dec_c = s_t_1
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()
        
        c_t_0 = torch.zeros((encoder_outputs.shape[0], config.hidden_dim * 2)).to(config.use_gpu)
        coverage_t_0 = torch.zeros([encoder_outputs.shape[0], encoder_outputs.shape[1]]).to(config.use_gpu) if config.is_coverage else None
        
        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[config.word2id["BOS"]],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]

        steps = 0
        results = []
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < len(config.id2word) else config.word2id["UNK"] for t in latest_tokens]
            y_t_1 = torch.tensor(latest_tokens).long().to(config.use_gpu)
            all_state_h = []
            all_state_c = []
            all_context = []
            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)
            
            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)
            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.seq2seqmodel.decoder(y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1, enc_batch_extend_vocab, extra_zeros, coverage_t_1, steps, op)

            # log_probs = torch.log(final_dist)
            # topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)
            topk_log_probs, topk_ids = torch.topk(final_dist, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()
            
            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)
                
                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)
            
            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == config.word2id["EOS"]:
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break
            
            steps += 1
        
        if len(results) == 0:
            results = beams
        
        beams_sorted = self.sort_beams(results)
        
        return beams_sorted[0].tokens
    
    def test_beam_decoder(self, args):
        self._load_model(args.model_name)
        self._makepath(args.model_name)
        self._getdata()
        self.seq2seqmodel.eval()
        print("[INFO] {} | test data batch length = {}".format(datetime.datetime.now(), len(self.test_iter)))
        print("predict [left, right) = [{}, {})".format(args.left, args.right))
        print("----"*10)
        print_epoch = len(self.test_iter) // 100 + 1
        with torch.no_grad():
            for idx, batch in enumerate(self.test_iter):
                if  args.left <= idx < args.right:
                    if idx % print_epoch == 0:
                        print("[INFO] {} | decoding process = {}".format(datetime.datetime.now(), idx))
                    best_summary = self.beam_search(batch)
                    output_ids = best_summary[1:]
                    decoded_words, ref_text = convert2word(output_ids, batch["oovs"][0]), self.reference[idx]["abstract"]
                    try:
                        fst_stop_idx = decoded_words.index("EOS")
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    write_for_rouge(ref_text, decoded_words, idx, self.ref_path, self.predict_path)
                # break
        print("[INFO] {} | the predition have finished ! ".format(datetime.datetime.now()))
        print("***" * 20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input process [left, right]")
    parser.add_argument("--left", type=int,default=0)
    parser.add_argument("--right", type=int, default=10)
    parser.add_argument("--cuda", default="cpu")
    parser.add_argument("--model_name",default="debug")
    parser.add_argument("--mode", default="cnndm-pg")

    args = parser.parse_args()

    config.use_gpu = args.cuda
    
    config.teacher_force = 2.0
    config.batch_size = 1
    test = TestSeq2Seq()
    test.test_beam_decoder(args)
    

