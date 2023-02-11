# sign language translation model with visual front
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import groupby
from sign2sql.modules.transformer import make_transformer_encoder, make_transformer_decoder, \
    PositionalEncoding, subsequent_mask, Embeddings
from sign2sql.modules.search import beam_search


class VisualFront(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, (1, 3, 3), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(16)
        self.conv1_activation = nn.PReLU(init=0)
        self.pool1 = nn.MaxPool3d((1, 3, 3), (1, 2, 2))
        
        self.conv2 = nn.Conv3d(16, 32, (1, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(32)
        self.conv2_activation = nn.PReLU(init=0)
        self.pool2 = nn.MaxPool3d((1, 3, 3), (1, 2, 2))

        self.conv31 = nn.Conv3d(32, 64, (1, 3, 3), padding=(0, 1, 1))
        self.bn31 = nn.BatchNorm3d(64)
        self.conv31_activation = nn.PReLU(init=0)
        self.conv32 = nn.Conv3d(64, 64, (1, 3, 3), padding=(0, 1, 1))
        self.bn32 = nn.BatchNorm3d(64)
        self.conv32_activation = nn.PReLU(init=0)
        self.pool3 = nn.MaxPool3d((1, 3, 3), (1, 2, 2))

        self.conv4 = nn.Conv3d(64, 128, (1, 3, 3), padding=(0, 1, 1))
        self.bn4 = nn.BatchNorm3d(128)
        self.conv4_activation = nn.PReLU(init=0)
        self.pool4 = nn.MaxPool3d((1, 3, 3), (1, 2, 2))

        self.fc5 = nn.Conv3d(128, 256, (1, 8, 8))
        self.bn5 = nn.BatchNorm3d(256)
        self.fc5_activation = nn.PReLU(init=0)

        self.fc6 = nn.Conv3d(256, d_model, (1, 1, 1))
        self.fc6_activation = nn.PReLU(init=0)
        self.init_params()
        
    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x:[bs, channels, d, h, w]; (h,w) == (144,144)
        x = self.conv1_activation(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.conv2_activation(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.conv31_activation(self.bn31(self.conv31(x)))
        x = self.conv32_activation(self.bn32(self.conv32(x)))
        x = self.pool3(x)
        x = self.conv4_activation(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.fc5_activation(self.bn5(self.fc5(x)))
        x = self.fc6_activation(self.fc6(x))
        x = x.squeeze(-1).squeeze(-1)
        return x


class SLTModel(nn.Module):
    def __init__(self, bert_word_embedding, d_model, dropout=0.1, num_layers=3, num_heads=8, vocab_size=30522):
        super().__init__()
        self.video_embedding = VisualFront(d_model)
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
        self.decoder = make_transformer_decoder(N_layer=num_layers,
                                                d_model=d_model,
                                                d_ff=d_model*4,
                                                heads=num_heads,
                                                dropout=dropout,
                                                ffn_layer='ffn',
                                                first_kernel_size=1)
        self.out_linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, video_array, video_array_mask, input_text_array, text_mask_array):
        x_embed = self.video_embedding(video_array.unsqueeze(1)).transpose(-1,-2)
        x_hidden = self.encoder(self.position_encoding(x_embed), mask=video_array_mask.unsqueeze(-2))
        
        y_embed = self.word_embedding(input_text_array)
        y_input_decoding_mask = text_mask_array.unsqueeze(-2) & subsequent_mask(text_mask_array.size(-1)).to(text_mask_array.device)
        y_hidden = self.decoder(self.position_encoding(y_embed), x_hidden,
                                src_mask=video_array_mask.unsqueeze(-2), tgt_mask=y_input_decoding_mask)
        y_pred = self.out_linear(y_hidden)
        return y_pred

        
def compute_ce_loss(y_pred, output_text_array, text_mask_array):
    ce_loss = torch.sum(nn.CrossEntropyLoss(reduction='none')(
                            y_pred.view(-1, y_pred.size(-1)), output_text_array.view(-1)
                            ) * text_mask_array.view(-1)
                        ) / torch.sum(text_mask_array)
    return ce_loss


def inference_slt_beam(slt_model, tokenizer, video_array, video_array_mask, beam_size=5, beam_max_length=200, alpha=0, vocab_size=30522):
    x_embed = slt_model.video_embedding(video_array.unsqueeze(1)).transpose(-1,-2)
    x_hidden = slt_model.encoder(slt_model.position_encoding(x_embed), mask=video_array_mask.unsqueeze(-2))

    stacked_txt_output, _ = beam_search(slt_model.word_embedding,
                                        slt_model.position_encoding,
                                        slt_model.decoder,
                                        slt_model.out_linear,
                                        beam_size,
                                        bos_index=101,  # [CLS] in vocab_uncased_L-12_H-768_A-12.txt
                                        eos_index=102,  # [SEP] in vocab_uncased_L-12_H-768_A-12.txt
                                        pad_index=0,
                                        encoder_output=x_hidden,
                                        encoder_hidden=None,
                                        src_mask=video_array_mask,
                                        max_output_length=beam_max_length,
                                        alpha=alpha,  # [-1,0,1,2,3,4,5]
                                        output_size=vocab_size,
                                        n_best=1)

    return stacked_txt_output

