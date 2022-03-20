"""Data utils.
"""

import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import random
import enum
import six
import copy
from six.moves import map
from six.moves import range
from six.moves import zip
import ast

import torch
from torch.autograd import Variable
import nltk
# Special vocabulary symbols
_PAD = "_PAD_" 
_EOS = "_EOS_" 
_GO = "_GO_" 
_UNK = "_UNK_" 
_START_VOCAB = [_PAD, _EOS, _GO, _UNK]

PAD_ID = 0
EOS_ID = 1
GO_ID = 2
UNK_ID = 3



def np_to_tensor(inp, output_type, cuda_flag):
    if output_type == 'float':
        inp_tensor = Variable(torch.FloatTensor(inp))
    elif output_type == 'int':
        inp_tensor = Variable(torch.LongTensor(inp))
    else:
        print('undefined tensor type')
    if cuda_flag:
        inp_tensor = inp_tensor.cuda()
    return inp_tensor


class DataProcessor(object):  
    def __init__(self, args):
        gloss_vocab={}
        with open(args.gloss_vocab) as f:
            for i,line in enumerate(f):
                gloss_vocab[line.split('\t')[0]]=i
        self.gloss_vocab = gloss_vocab
        sql_vocab={}
        with open(args.sql_vocab) as f:
            for i,line in enumerate(f):
                sql_vocab[line.split('\t')[0]]=i
        self.sql_vocab = sql_vocab
        self.gloss_vocab_list = _START_VOCAB[:]
        self.sql_vocab_list = _START_VOCAB[:]
        self.vocab_offset = len(_START_VOCAB)
        for word in self.gloss_vocab:
            while self.gloss_vocab[word] + self.vocab_offset >= len(self.gloss_vocab_list):
                self.gloss_vocab_list.append(word)
            # self.gloss_vocab_list[self.gloss_vocab[word] + self.vocab_offset] = word
        for word in self.sql_vocab:
            while self.sql_vocab[word] + self.vocab_offset >= len(self.sql_vocab_list):
                self.sql_vocab_list.append(word)
            self.sql_vocab_list[self.sql_vocab[word] + self.vocab_offset] = word
        self.gloss_vocab_size = len(self.gloss_vocab) + self.vocab_offset
        self.sql_vocab_size = len(self.sql_vocab) + self.vocab_offset
        self.cuda_flag = args.cuda
        # self.target_code_transform = args.target_code_transform
        # self.hierarchy = args.hierarchy
        # self.copy_mechanism = args.copy_mechanism
        # self.max_num_code_cells = args.max_num_code_cells
        self.max_gloss_len = args.max_gloss_len
        # self.max_code_context_len = args.max_code_context_len
        self.max_decode_len = args.max_decode_len

    def ids_to_prog(self,ids):
        prog = []
        for i in ids:
            if int(i) < self.sql_vocab_size:
                prog += [self.sql_vocab_list[int(i)]]
            if int(i) == EOS_ID:
                break
        return prog

    def get_joint_plot_type(self, init_label):
        if init_label in [0, 3]:
            return init_label
        elif init_label in [1, 4]:
            return 1
        else:
            return 2

    def load_data(self, gloss_file, sql_file,sql_mask):
        gloss_samples = []
        sql_samples = []
        with open(gloss_file) as f :
            for i,line in enumerate(f):
                gloss_samples.append(line.strip())
        with open(sql_file) as f:
            for i,line in enumerate(f):
                sql_samples.append(line.strip())
        mask = []
        with open(sql_mask) as f:
            for i, line in enumerate(f):
                line = line.strip().split(' ')
                mask.append([int(i) for i in line])

        return gloss_samples,sql_samples,mask

    def parser(self,sentence):
        tokens = nltk.word_tokenize(sentence.lower())
        return tokens
    def preprocess(self, gloss_samples,sql_samples,sql_mask):
        data = []
        indices = []
        max_target_code_seq_len = 0
        min_target_code_seq_len = 512
        for sample_idx, sample in enumerate(zip(gloss_samples,sql_samples,sql_mask)):
            target_code_seq = sample[1].split(' ')
            nl = sample[0].split(' ')
            mask=sample[2]
            nl=nl[:self.max_gloss_len-1]
            # target_code_seq = sample[1]
            # nl = sample[0]
            max_target_code_seq_len = max(max_target_code_seq_len, len(target_code_seq))
            min_target_code_seq_len = min(min_target_code_seq_len, len(target_code_seq))

            input_word_seq = []

            for word in nl:
                if word in self.gloss_vocab:
                    input_word_seq.append(self.gloss_vocab[word] + self.vocab_offset)
                else:
                    input_word_seq.append(UNK_ID)

            output_code_seq = []
            output_code_seq+=[GO_ID]

            for i, tok in enumerate(target_code_seq):
                if tok in self.sql_vocab:
                    output_code_seq.append(self.sql_vocab[tok] + self.vocab_offset)
                else:
                    output_code_seq.append(UNK_ID)

            input_word_seq += [EOS_ID]
            output_code_seq += [EOS_ID]

            cur_data = {}

            cur_data['input_word_seq'] = input_word_seq
            cur_data['output_code_seq'] = output_code_seq
            cur_data['output_mask']=mask

            data.append(cur_data)
            indices.append(sample_idx)
        print('code seq len: min: ', min_target_code_seq_len, 'max: ', max_target_code_seq_len)
        return data, indices

    def get_batch(self, data, batch_size, start_idx):
        data_size = len(data)


        batch_word_input = []


        batch_code_output = []

        batch_code_output_mask = []
        max_word_len = 0

        max_output_code_len = 0

        for idx in range(start_idx, min(start_idx + batch_size, data_size)):
            cur_sample = data[idx]

            batch_word_input.append(cur_sample['input_word_seq'])
            max_word_len = max(max_word_len, len(cur_sample['input_word_seq']))


            batch_code_output.append(cur_sample['output_code_seq'])
            max_output_code_len = max(max_output_code_len, len(cur_sample['output_code_seq']))

            batch_code_output_mask.append(cur_sample['output_mask'])

        for idx in range(len(batch_word_input)):
            if len(batch_word_input[idx]) < max_word_len:
                batch_word_input[idx] = batch_word_input[idx] + [PAD_ID] * (max_word_len - len(batch_word_input[idx]))
        batch_word_input = np.array(batch_word_input)
        batch_word_input = np_to_tensor(batch_word_input, 'int', self.cuda_flag)

        for idx in range(len(batch_code_output)):
            if len(batch_code_output[idx]) < max_output_code_len:
                batch_code_output[idx] = batch_code_output[idx] + [PAD_ID] * (
                            max_output_code_len - len(batch_code_output[idx]))

        batch_code_output = np.array(batch_code_output)
        batch_code_output = np_to_tensor(batch_code_output, 'int', self.cuda_flag)

        return batch_word_input,batch_code_output,batch_code_output_mask

