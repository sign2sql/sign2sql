# sign2sql_v2 using pretrained video embedding from SLT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import groupby
from sign2sql.modules.transformer import make_transformer_encoder, make_transformer_decoder, \
    PositionalEncoding, subsequent_mask, Embeddings
from sqlova.utils.utils_wikisql import get_wemb_h
from sign2sql.modules.search import beam_search

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------- utils_sign2sql_encoder ---------------
def generate_sign2sql_headers_inputs(tokenizer, hds1):
    tokens = []
    tokens.append("[SEP]")
    i_hds = []
    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_hds.append((i_st_hd, i_ed_hd))
        tokens.append("[SEP]")
    return tokens, i_hds

def gen_sign2sql_l_hpu(i_hds):
    """
    # Treat columns as if it is a batch of natural language utterance with batch-size = # of columns * # of batch_size
    i_hds = [(17, 18), (19, 21), (22, 23), (24, 25), (26, 29), (30, 34)])
    """
    l_hpu = []
    for i_hds1 in i_hds:
        for i_hds11 in i_hds1:
            l_hpu.append(i_hds11[1] - i_hds11[0])

    return l_hpu

def get_sign2sql_encoder_output(sign2sql_encoder, tokenizer, video_array, video_array_mask, hds, 
                                input_where_array, where_mask_array):
    # hds: list of list of string
    l_n = []  # length of subtokens for each nlu (valid length of each video)
    l_hs = []  # The length of columns for each batch

    l_n = torch.sum(video_array_mask, dim=1).cpu().tolist()
    l_hs = [len(hds1) for hds1 in hds]

    input_ids = []
    input_mask = []
    i_hds = []
    input_ids_lens = []
    for b, hds1 in enumerate(hds):
        tokens1, i_hds1 = generate_sign2sql_headers_inputs(tokenizer, hds1)
        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        input_ids.append(input_ids1)
        input_ids_lens.append(len(input_ids1))
        i_hds.append(i_hds1)
    
    max_seq_length = max(input_ids_lens)
    # padding
    for b in range(len(hds)):
        input_ids1 = input_ids[b]

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        # 3. Zero-pad up to the sequence length.
        while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)
        
        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length

        input_mask.append(input_mask1)
    
    # Convert to tensor
    header_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    header_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)

    # 4. Generate Sign2SQL Encoder output.
    encoder_output, y_pred = sign2sql_encoder(video_array, video_array_mask, 
                                            header_input_ids, header_input_mask,
                                            input_where_array, where_mask_array)

    # 5. generate l_hpu from i_hds
    l_hpu = gen_sign2sql_l_hpu(i_hds)

    return encoder_output, y_pred, l_n, l_hpu, l_hs, i_hds, header_input_ids, header_input_mask


def get_sign2sql_wemb_video(encoder_output, batch_video_max_len):
    return encoder_output[:, :batch_video_max_len]

def get_sign2sql_wemb_header(encoder_output, batch_video_max_len, i_hds, l_hpu, l_hs):
    l_hpu_max = max(l_hpu)
    num_of_all_hds = sum(l_hs)
    wemb_h = torch.zeros([num_of_all_hds, l_hpu_max, encoder_output.size(-1)]).to(device)
    b_pu = -1
    for b, i_hds1 in enumerate(i_hds):
        for b1, i_hds11 in enumerate(i_hds1):
            b_pu += 1
            wemb_h[b_pu, 0:(i_hds11[1] - i_hds11[0])] = encoder_output[b, batch_video_max_len+i_hds11[0]:batch_video_max_len+i_hds11[1]]
    
    return wemb_h


def get_wemb_sign2sql_encoder(sign2sql_encoder, tokenizer, video_array, video_array_mask, 
                            hds, input_where_array, where_mask_array):
    encoder_output, y_pred, l_n, l_hpu, l_hs, i_hds, header_input_ids, header_input_mask = get_sign2sql_encoder_output(sign2sql_encoder, tokenizer, video_array, video_array_mask, 
                                                                        hds, input_where_array, where_mask_array)
    
    # get the wemb
    wemb_v = get_sign2sql_wemb_video(encoder_output, video_array_mask.size(1))
    
    wemb_h = get_sign2sql_wemb_header(encoder_output, video_array_mask.size(1), i_hds, l_hpu, l_hs)
    
    return y_pred, wemb_v, wemb_h, l_n, l_hpu, l_hs, header_input_ids, header_input_mask


# --------------- sign2sql_v2 model ---------------

class Sign2SQLv2Model(nn.Module):
    def __init__(self, pretrained_slt_model, bert_word_embedding, d_model, out_d_model, dropout=0.1, num_layers=3, num_heads=8, vocab_size=30522):
        super().__init__()
        self.pretrained_slt_model = pretrained_slt_model
        for param in self.pretrained_slt_model.parameters():
            param.requires_grad = False
        self.video_embedding = nn.Linear(d_model, d_model, bias=False)
        self.position_encoding = PositionalEncoding(d_model, dropout)
        self.encoder = make_transformer_encoder(N_layer=num_layers,
                                                d_model=d_model,
                                                d_ff=d_model*4,
                                                heads=num_heads,
                                                dropout=dropout,
                                                ffn_layer='ffn',
                                                first_kernel_size=1)
        self.bert_word_embedding = bert_word_embedding  # type BERTEmbeddings
        bert_hidden_size = 768  # from bert_config_uncased_L-12_H-768_A-12.json
        for param in self.bert_word_embedding.parameters():
            param.requires_grad = False
        self.word_embedding = nn.Sequential(
            self.bert_word_embedding.word_embeddings,
            nn.Linear(bert_hidden_size, d_model, bias=False)
        )
        self.segment_embedding = nn.Embedding(2, d_model)
        self.encoder_out_linear = nn.Linear(d_model, out_d_model)
        # where decoder
        self.decoder = make_transformer_decoder(N_layer=num_layers,
                                                d_model=d_model,
                                                d_ff=d_model*4,
                                                heads=num_heads,
                                                dropout=dropout,
                                                ffn_layer='ffn',
                                                first_kernel_size=1)
        self.where_out_linear = nn.Linear(d_model, vocab_size)
    
    def gen_segments(self, l_seg1, l_seg2, device):
        seg_ids = [1] * l_seg1 + [0] * l_seg2
        seg_ids = torch.tensor(seg_ids, dtype=torch.long).to(device)
        return seg_ids

    def concat_video_header(self, x1_embed, x1_mask, x2_embed, x2_mask):
        x_embed = torch.cat((x1_embed, x2_embed), dim=1)
        x_mask = torch.cat((x1_mask, x2_mask), dim=1)
        return x_embed, x_mask

    def forward(self, video_array, video_array_mask, header_input_ids, header_input_mask,
                where_input_ids, where_ids_mask):
        # where_input_ids: [b, T]: [[wc2, wo2, wv2], [wc2, wo2, wv2]...]
        x1_embed = self.pretrained_slt_model.video_embedding(video_array.unsqueeze(1)).transpose(-1,-2)
        x1_embed = self.pretrained_slt_model.encoder(self.pretrained_slt_model.position_encoding(x1_embed), mask=video_array_mask.unsqueeze(-2))
        x1_embed = self.video_embedding(x1_embed)
        x2_embed = self.word_embedding(header_input_ids)
        
        x_embed, x_mask = self.concat_video_header(x1_embed, video_array_mask, x2_embed, header_input_mask)
        seg_ids = self.gen_segments(x1_embed.size(1), x2_embed.size(1), x_embed.device)

        x_embed = x_embed + self.segment_embedding(seg_ids.unsqueeze(0)) 
        x_hidden = self.encoder(self.position_encoding(x_embed), mask=x_mask.unsqueeze(-2))
        
        x_output = self.encoder_out_linear(x_hidden)  # send to SQLova decoding model for sc, sa, wn

        # where decoding
        y_embed = self.word_embedding(where_input_ids)
        y_input_decoding_mask = where_ids_mask.unsqueeze(-2) & subsequent_mask(where_ids_mask.size(-1)).to(where_ids_mask.device)
        y_hidden = self.decoder(self.position_encoding(y_embed), x_hidden,
                                src_mask=x_mask.unsqueeze(-2), tgt_mask=y_input_decoding_mask)
        y_pred = self.where_out_linear(y_hidden)
        return x_output, y_pred


def compute_ce_loss(y_pred, where_output_ids, where_ids_mask):
    ce_loss = torch.sum(nn.CrossEntropyLoss(reduction='none')(
                            y_pred.view(-1, y_pred.size(-1)), where_output_ids.view(-1)
                            ) * where_ids_mask.view(-1)
                        ) / torch.sum(where_ids_mask)
    return ce_loss


def inference_where_beam(sign2sql_model, tokenizer, 
                         video_array, video_array_mask, header_input_ids, header_input_mask, 
                         beam_size=5, beam_max_length=200, alpha=0, vocab_size=30522):
    x1_embed = sign2sql_model.pretrained_slt_model.video_embedding(video_array.unsqueeze(1)).transpose(-1,-2)
    x1_embed = sign2sql_model.pretrained_slt_model.encoder(sign2sql_model.pretrained_slt_model.position_encoding(x1_embed), mask=video_array_mask.unsqueeze(-2))
    x1_embed = sign2sql_model.video_embedding(x1_embed)
    x2_embed = sign2sql_model.word_embedding(header_input_ids)
    
    x_embed, x_mask = sign2sql_model.concat_video_header(x1_embed, video_array_mask, x2_embed, header_input_mask)
    seg_ids = sign2sql_model.gen_segments(x1_embed.size(1), x2_embed.size(1), x_embed.device)

    x_embed = x_embed + sign2sql_model.segment_embedding(seg_ids.unsqueeze(0))
    x_hidden = sign2sql_model.encoder(sign2sql_model.position_encoding(x_embed), mask=x_mask.unsqueeze(-2))

    stacked_txt_output, _ = beam_search(sign2sql_model.word_embedding,
                                        sign2sql_model.position_encoding,
                                        sign2sql_model.decoder,
                                        sign2sql_model.where_out_linear,
                                        beam_size,
                                        bos_index=101,  # [CLS] in vocab_uncased_L-12_H-768_A-12.txt
                                        eos_index=102,  # [SEP] in vocab_uncased_L-12_H-768_A-12.txt
                                        pad_index=0,
                                        encoder_output=x_hidden,
                                        encoder_hidden=None,
                                        src_mask=x_mask,
                                        max_output_length=beam_max_length,
                                        alpha=alpha,  # [-1,0,1,2,3,4,5]
                                        output_size=vocab_size,
                                        n_best=1)

    return stacked_txt_output

