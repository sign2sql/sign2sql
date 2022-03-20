import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import cuda
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import torch.nn.functional as F

import numpy as np
from .data_utils import data_utils
import random
from d2l import torch as d2l


class Encoder(nn.Module):
    # src_vocab_size德语词汇表大小，emb_model词向量维度，hidden_size隐藏向量维度，n_layers lstm深度
    def __init__(self,args):

        self.gloss_vocab_size=args.gloss_vocab_size
        self.embedding_size=args.embedding_size
        self.LSTM_hidden_size=args.LSTM_hidden_size
        self.num_LSTM_layers=args.num_LSTM_layers
        self.dropout_rate=args.dropout_rate
        self.bidirectional=args.bidirectional

        super(Encoder, self).__init__()

        self.gloss_embedding = nn.Embedding(self.gloss_vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.LSTM_hidden_size, num_layers=self.num_LSTM_layers, batch_first=True,
                            dropout=self.dropout_rate,bidirectional=self.bidirectional)

        if self.bidirectional:
            self.fc1=nn.Linear(self.LSTM_hidden_size*2,self.LSTM_hidden_size)
            self.fc2=nn.Linear(self.LSTM_hidden_size*2,self.LSTM_hidden_size)

    def forward(self, batch_input):
        # h[0]：第一层正向最后时间步隐藏状态
        # h[1]：第一层反向最后时间步隐藏状态
        # h[2]：第二层正向最后时间步隐藏状态
        # h[3]：第二层反向最后时间步隐藏状态

        # src[batch_size,seq_len]
        batch_input_embedding = self.gloss_embedding(batch_input)
        # src[batch_size,seq_len,emb_model]
        output, (h_n, c_n) = self.lstm(batch_input_embedding)
        # output[batch_size,seq_len,hidden_size]  最后一层每个时间步的隐状态h_t

        # h_n[n_layers,batch_size,hidden_size] 最后一个时间步每层的隐状态(实际上并非这样，Pytorch机制原因)
        # c_n[n_layers,batch_size,hidden_size] 最后一个时间步每层的记忆c（实际上并非这样，Pytorch机制原因）
        if self.bidirectional:
            if self.num_LSTM_layers >= 2:
                h_n = torch.tanh(self.fc1(
                    torch.cat((h_n[0:self.num_LSTM_layers, :, :], h_n[self.num_LSTM_layers:self.num_LSTM_layers * 2, :, :]), dim=2)))
                c_n = torch.tanh(self.fc2(
                    torch.cat((c_n[0:self.num_LSTM_layers, :, :], c_n[self.num_LSTM_layers:self.num_LSTM_layers * 2, :, :]), dim=2)))
            else:
                h_n = torch.tanh(self.fc1(torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=1))).unsqueeze(0)
                c_n = torch.tanh(self.fc2(torch.cat((c_n[0, :, :], c_n[1, :, :]), dim=1))).unsqueeze(0)

        return output, (h_n, c_n)  # output的意义不大，主要是(h_n,c_n)，其作为上下文向量


class Attention(nn.Module):
    def __init__(self, args):  # 即encoder和decoder的输出维度
        super(Attention, self).__init__()
        self.LSTM_hidden_size = args.LSTM_hidden_size
        self.bidirectional=args.bidirectional
        # 双向的话，enc_hid_dim要乘2
        if self.bidirectional:
            self.w = nn.Linear((self.LSTM_hidden_size * 2) + self.LSTM_hidden_size, self.LSTM_hidden_size, bias=False)
        else:
            self.w = nn.Linear(self.LSTM_hidden_size + self.LSTM_hidden_size, self.LSTM_hidden_size, bias=False)
        self.v = nn.Linear(self.LSTM_hidden_size, 1, bias=False)  # 不要偏置，做一个线性变换

    def forward(self, h, enc_out):
        # h = [n_layers,batch_size, dec_hid_dim]
        # enc_out = [batch_size,src_len, enc_hid_dim]
        h = h.squeeze(0)
        src_len = enc_out.shape[1]
        # 因为要将当前时刻的h和所有时刻的h即enc_out（一个src_len个时刻）的h拼接送入全连接，所以先将h复制src_len次，然后拼接
        if len(h.shape) == 3:  h = h[0]  # torch.Size([2, 16, 512])   #当lstm多层时，只取最后一层的h。
        h = h.unsqueeze(1).repeat(1, src_len, 1) #(batch_size,src_len, lstm_hidden_dim)

        # energy = [batch_size, src_len, lstm_hidden_dim]
        energy = torch.tanh(self.w(torch.cat((h, enc_out), dim=2)))

        # 因为经过self.v后维度变为：[batch_size, src_len, 1]，所以squeeze掉最后一个维度
        attention = self.v(energy).squeeze(2)

        # 用softmax在src_len维度归一化一波，就得到了attention权重
        return F.softmax(attention, dim=1)


class ATT_Decoder(nn.Module):
    # trg_vocab_size 目标端的词汇表大小
    # emb_dim为词向量维度（我们将其设置与源端一样大小）
    # hidden_size 为目标端隐层维度（将其设置为与源端一样大小）
    # n_layers 网络层数（将其设置为一样大小）
    def __init__(self,args,attention):
        super(ATT_Decoder, self).__init__()
        self.sql_vocab_size = args.sql_vocab_size
        self.embedding_size = args.embedding_size
        self.LSTM_hidden_size = args.LSTM_hidden_size
        self.num_LSTM_layers = args.num_LSTM_layers
        self.dropout_rate = args.dropout_rate
        self.args=args
        self.attention=attention
        self.bidirectional=args.bidirectional

        self.emb = nn.Embedding(self.sql_vocab_size, self.embedding_size)
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_size*2+self.LSTM_hidden_size,self.LSTM_hidden_size,num_layers=self.num_LSTM_layers,
                                dropout=self.dropout_rate,batch_first=True)
            self.classify=nn.Linear(self.LSTM_hidden_size*2+self.LSTM_hidden_size+self.embedding_size, self.sql_vocab_size)
        else:
            self.lstm = nn.LSTM(self.embedding_sizeself.LSTM_hidden_size, self.LSTM_hidden_size,
                                num_layers=self.num_LSTM_layers, dropout=self.dropout_rate,batch_first=True)
            self.classify = nn.Linear(self.LSTM_hidden_size + self.LSTM_hidden_size + self.embedding_size,
                                      self.sql_vocab_size)
        # self.liner=nn.Linear(self.LSTM_hidden_size*2,self.embedding_size)

    def forward(self, decoder_input, encoder_outputs,h_n, c_n):
        # trg为应该为[batch,seq_len,dim]，不过实际训练中是一个一个传入（要考虑是否采用强制教学），所以seq_len为1
        # trg真正的输入维度为[batch]
        # h_n与c_n是源端的上下文向量（若计算不指定，则默认为0（若Encoder编码中））
        # 维度均为：[n_layers,batch_size,hidden_size]
        dec_input = decoder_input.unsqueeze(1) #dec_input = [batch_size, 1]
        # trg[batch,1,dim]
        trg = self.emb(dec_input)

        # 用decoder的当前状态h和encoder的所有时刻的状态enc_output计算注意力权重。
        att = self.attention(h_n,encoder_outputs).unsqueeze(1)  # 多增加一个维度，变成：a = [batch_size, 1, src_len]

        # att和enc_output矩阵相乘相当于维度变为:batch_size  x  (1 x src_len)  x  (src_len x enc_hid_dim) 变为[batch_size, 1, enc_hid_dim]
        att_c = torch.bmm(att, encoder_outputs)

        lstm_input = torch.cat((trg, att_c), dim=2)  # lstm_input = [batch_size, 1,lstm_hid_dim + emb_dim]

        dec_output, (dec_h, dec_c) = self.lstm(lstm_input, (h_n, c_n))

        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # att_c = [batch_size, enc_hid_dim]
        embedded = trg.squeeze(1)
        dec_output = dec_output.squeeze(1)
        att_c = att_c.squeeze(1)
        # 把这3者拼接，然后送入最后的全连接层分类，pred = [batch_size, output_dim]

        pred = self.classify(torch.cat((dec_output, att_c, embedded), dim=1))

        return pred, (dec_h, dec_c)  # 返回(h_n,c_n)是为了下一解码器继续使用
class Decoder(nn.Module):
    # trg_vocab_size 目标端的词汇表大小
    # emb_dim为词向量维度（我们将其设置与源端一样大小）
    # hidden_size 为目标端隐层维度（将其设置为与源端一样大小）
    # n_layers 网络层数（将其设置为一样大小）
    def __init__(self,args):
        super(Decoder, self).__init__()
        self.sql_vocab_size = args.sql_vocab_size
        self.embedding_size = args.embedding_size
        self.LSTM_hidden_size = args.LSTM_hidden_size
        self.num_LSTM_layers = args.num_LSTM_layers
        self.dropout_rate = args.dropout_rate
        self.args=args
        self.bidirectional=args.bidirectional

        self.emb = nn.Embedding(self.sql_vocab_size, self.embedding_size)

        self.lstm = nn.LSTM(self.embedding_size, self.LSTM_hidden_size, num_layers=self.num_LSTM_layers,
                            batch_first=True, dropout=self.dropout_rate)
        self.classify = nn.Linear(self.LSTM_hidden_size,self.sql_vocab_size)
        # self.liner=nn.Linear(self.LSTM_hidden_size*2,self.embedding_size)

    def forward(self, decoder_input, encoder_outputs,h_n, c_n):
        # trg为应该为[batch,seq_len,dim]，不过实际训练中是一个一个传入（要考虑是否采用强制教学），所以seq_len为1
        # trg真正的输入维度为[batch]
        # h_n与c_n是源端的上下文向量（若计算不指定，则默认为0（若Encoder编码中））
        # 维度均为：[n_layers,batch_size,hidden_size]
        dec_input = decoder_input.unsqueeze(1) #dec_input = [batch_size, 1]
        # trg[batch,1,dim]
        trg = self.emb(dec_input)

        dec_output, (dec_h, dec_c) = self.lstm(trg, (h_n, c_n))
        pred=self.classify(dec_output.squeeze())

        return pred, (dec_h, dec_c)  # 返回(h_n,c_n)是为了下一解码器继续使用

class Seq2Seq(nn.Module):
    def __init__(self, args,encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sql_vocab_size=args.sql_vocab_size
        self.lr=args.lr
        self.cuda_flag=args.cuda
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        elif args.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            raise ValueError('optimizer undefined: ', args.optimizer)
        self.criteration =nn.CrossEntropyLoss(ignore_index=data_utils.PAD_ID)

    def init_weights(self, param_init):
        for param in self.parameters():
            nn.init.uniform_(param, -param_init, param_init)
    def forward(self, batch_input, batch_target,batch_target_mask, teach_rate=0.5):
        # src [bacth,seq_len]
        # trg  [bacth,seq_len]
        # teach_radio 强制教学的阈值
        batch_size =batch_target.shape[0]
        trg_seqlen = batch_target.shape[1]
        # 保存每次输出的结果
        outputs_save = torch.zeros(batch_size, trg_seqlen, self.sql_vocab_size)
        output_prediction = torch.zeros(batch_size, trg_seqlen,dtype=torch.int64)
        # 第一个输入到解码器中为<sos>
        trg_i = torch.ones(batch_size, dtype=torch.int64) * data_utils.GO_ID
        output_prediction[:,0]=trg_i
        target_mask=torch.tensor(batch_target_mask,dtype=torch.float)
        if (cuda):
            outputs_save = outputs_save.cuda()
            output_prediction=output_prediction.cuda()
            trg_i=trg_i.cuda()
            target_mask=target_mask.cuda()
        # 对源端进行编码
        encoder_outputs, (h_n, c_n) = self.encoder(batch_input)

        # trg_i = batch_target[:,1]
        # trg_i [batch]
        for i in range(1,trg_seqlen):
            output, (h_n, c_n) = self.decoder(trg_i,encoder_outputs, h_n, c_n)
            # output[batch trg_vocab_size]
            # outputs_save[:, i, :] = output
            if target_mask.shape[1] !=1:
                output=output.mul(target_mask)
            outputs_save[:, i, :] = output
            output_prediction [:,i]= output.max(1)[1]
            # 产生一个随机概率(即是否强制教学)
            probability = random.random()

            # 获取时间步预测的结果
            top = output.argmax(1)
            # top[batch]
            # 下一时间步的输入
            trg_i = batch_target[:, i] if probability > teach_rate else top
        return outputs_save,output_prediction