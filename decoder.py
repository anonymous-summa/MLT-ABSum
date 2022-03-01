# _*_coding: utf-8_*_

from torch.nn.utils.rnn import pad_sequence
from torch.nn.init import xavier_uniform_
import config as config
from encoder import *
import torch
import torch.nn as nn
import sys

def init_pos_normal(wt):
    wt.data.normal_(0,0.1) # position embedding

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        # B * t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k
        attn_dist_ = torch.softmax(scores, dim=1) * enc_padding_mask  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        
        attn_dist = attn_dist_ / (normalization_factor.view(-1, 1) + torch.ones_like(normalization_factor.view(-1, 1)) * sys.float_info.epsilon)

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()
        # original inputs
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        # pos embedding
        self.pos_embedding = nn.Embedding(len(config.pos2id), config.emb_dim)
        # dp embedding
        self.dp_results_embedding = nn.Embedding(len(config.relation2id), config.emb_dim)


        # T.S lstm
        self.lstm = nn.LSTM(config.hidden_dim*2+config.emb_dim*2, config.hidden_dim, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        # reduce hidden
        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

        # features
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)

    def get_decoder_init(self, hidden):
        h, c = hidden
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = torch.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = torch.relu(self.reduce_c(c_in))
        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))

    def cross_task_reduce(self, seq2seq_encoder_outputs, class_outputs, dp_outputs, pos_outputs):

        batch_size, seq_len, hidden_dim = pos_outputs.shape

        cross_q = seq2seq_encoder_outputs.unsqueeze(2)

        combine_hidden = torch.cat([seq2seq_encoder_outputs, class_outputs, dp_outputs, pos_outputs], dim=2).view(batch_size, seq_len, -1, hidden_dim)

        atten = torch.matmul(cross_q, combine_hidden.transpose(2, 3))
        
        atten = torch.softmax(atten, dim=3)
        cross_atten = torch.matmul(atten, combine_hidden).squeeze(2)

        encoder_outputs = cross_atten.squeeze(2).contiguous()
        encoder_features = self.W_h(encoder_outputs.view(-1, hidden_dim))

        return encoder_outputs, encoder_features

    def get_dp_label_embedding(self, batch_heads, batch_relations):
        batch_arc_embeded = []
        for i in range(len(batch_heads)): # item of the batch
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

    def forward(self, inputs_text, class_outputs, class_hidden, dp_outputs, dp_heads, dp_relations, pos_outputs, pos_results):
        word_embedded = self.embedding(inputs_text)  # [batch,seq_len,embed_dim]

        arc_embedded = self.get_dp_label_embedding(dp_heads, dp_relations)

        pos_embedded = self.pos_embedding(pos_results)

        embedded = torch.cat((class_outputs, word_embedded, arc_embedded+pos_embedded),dim=2)  # [batch,seq_len,embed_dim]

        seq2seq_encoder_outputs, hidden = self.lstm(embedded, class_hidden)

        init_decoder_hidden = self.get_decoder_init(hidden)

        final_hidden, final_features = self.cross_task_reduce(seq2seq_encoder_outputs, class_outputs, dp_outputs, pos_outputs)

        return final_hidden, final_features, init_decoder_hidden



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()

        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        
        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1, enc_batch_extend_vocab, extra_zeros, coverage, step, op="train"):

        # if not self.training and step == 0:
        #     h_decoder, c_decoder = s_t_1
        #     s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim), c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        #     c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage)
        #     coverage = coverage_next

        y_t_1_embed = self.embedding(y_t_1)  # decoder inputs

        x = self.x_context(torch.cat([c_t_1, y_t_1_embed], dim=1))

        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim), c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage)

        # if self.training or step > 0:
        coverage = coverage_next
        
        # c_t means the context of the current words about inputs sequence
        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)  # B x hidden_dim * 3

        output = self.out1(output) # TODO: differences
        output = self.out2(output)

        vocab_dist = torch.softmax(output, dim=1)  # B x hidden_dim  -> B x vocab_size

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)    # TODO miss
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist
            
            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros],dim=1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage
