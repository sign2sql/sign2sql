import os, json
import random as rd
from copy import deepcopy

from matplotlib.pylab import *

import torch
import torchvision.datasets as dsets
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data -----------------------------------------------------------------------------------------------
def load_sign2sql_test(path_sign2sql):
    # Get data
    test_data = load_sign2text_data(path_sign2sql, mode='test')

    # Get table
    path_table = os.path.join(path_sign2sql, 'all.tables.jsonl')
    table = {}
    with open(path_table) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            table[t1['id']] = t1
    
    return test_data, table

def get_loader_sign2sql_test(data_test, bS, shuffle_test=False):
    test_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_test,
        shuffle=shuffle_test,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return test_loader


# Load data -----------------------------------------------------------------------------------------------
def load_sign2text(path_sign2text):
    # Get data
    train_data = load_sign2text_data(path_sign2text, mode='train')
    dev_data = load_sign2text_data(path_sign2text, mode='dev')

    # Get table
    path_table = os.path.join(path_sign2text, 'all.tables.jsonl')
    table = {}
    with open(path_table) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            table[t1['id']] = t1
    
    return train_data, dev_data, table

def load_sign2text_data(path_sign2text, mode='train'):
    path_sql = os.path.join(path_sign2text, mode+'_tok.jsonl')
    data = []
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            if os.path.exists(t1['video_path']):
                data.append(t1)
    return data

def get_loader_sign2text(data_train, data_dev, bS, shuffle_train=True, shuffle_dev=False):
    train_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_train,
        shuffle=shuffle_train,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    dev_loader = torch.utils.data.DataLoader(
        batch_size=bS,
        dataset=data_dev,
        shuffle=shuffle_dev,
        num_workers=4,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_loader, dev_loader

def get_fields_sign2text_1(t1):
    nlu1 = t1['question']
    vid_path = t1['video_path']
    video = np.load(vid_path)
    return nlu1, video

def get_fields_sign2text(t1s):
    nlu = []
    videos = []
    for t1 in t1s:
        nlu1, video = get_fields_sign2text_1(t1)
        
        nlu.append(nlu1)
        videos.append(video)
    return nlu, videos

def get_padded_batch_video(videos, max_vid_len=150):
    # sample rate 4 or 5 frame
    bS = len(videos)
    video_downsampled = []
    vid_lens = []
    vid_shape = None
    for vid in videos:
        tmp = vid[::5]  # or 4
        video_downsampled.append(tmp)
        vid_lens.append(tmp.shape[0])
        if vid_shape is None:
            vid_shape = (tmp.shape[1], tmp.shape[2])
    video_array = np.zeros([bS, min(max(vid_lens), max_vid_len), vid_shape[0], vid_shape[1]])
    video_array_mask = np.zeros([bS, min(max(vid_lens), max_vid_len)])
    for b in range(bS):
        video_array[b, :min(vid_lens[b], max_vid_len)] = video_downsampled[b][:max_vid_len]
        video_array_mask[b, :min(vid_lens[b], max_vid_len)] = 1
    video_array = torch.from_numpy(video_array).type(torch.float32).to(device)
    video_array_mask = torch.from_numpy(video_array_mask==1).to(device)
    return video_array, video_array_mask

def get_input_output_token(tokenizer, nlu):
    # tokenizer: BERT tokenizer (subword)
    bS = len(nlu)
    input_text = []
    output_text = []
    text_lens = []
    for nlu1 in nlu:
        tokens = []
        tokens.append("[CLS]")
        tokens += tokenizer.tokenize(nlu1)
        tokens.append("[SEP]")
        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_text.append(ids[:-1])
        output_text.append(ids[1:])
        text_lens.append(len(ids)-1)
    input_text_array = []
    output_text_array = []
    text_mask_array = np.zeros([bS, max(text_lens)])
    for b in range(bS):
        input_text_array.append(
            input_text[b] + [0] * (max(text_lens) - text_lens[b])
        )
        output_text_array.append(
            output_text[b] + [0] * (max(text_lens) - text_lens[b])
        )
        text_mask_array[b, :text_lens[b]] = 1
    input_text_array = torch.tensor(input_text_array, dtype=torch.long, device=device)
    output_text_array = torch.tensor(output_text_array, dtype=torch.long, device=device)
    text_mask_array = torch.from_numpy(text_mask_array==1).to(device)
    return input_text_array, output_text_array, text_mask_array


def get_input_output_where_ids(tokenizer, sql):
    # sql: list of dict [{'conds':[[2, 0, 'ABC ABC'],[...]]}, {}]
    bS = len(sql)
    input_where_ids = []
    output_where_ids = []
    where_ids_lens = []
    for sql_i in sql:
        tokens = []
        conds = sql_i['conds']
        for cond in conds:
            tokens.append("[CLS]")
            tokens.append(str(cond[0]))
            tokens.append(str(cond[1]))
            tokens += tokenizer.tokenize(str(cond[2]))
        if len(tokens) == 0:
            tokens.append("[CLS]")  # no cond found!
        tokens.append("[SEP]")

        ids = tokenizer.convert_tokens_to_ids(tokens)
        input_where_ids.append(ids[:-1])
        output_where_ids.append(ids[1:])
        where_ids_lens.append(len(ids)-1)
    
    input_where_array = []
    output_where_array = []
    where_mask_array = np.zeros([bS, max(where_ids_lens)])
    for b in range(bS):
        input_where_array.append(
            input_where_ids[b] + [0] * (max(where_ids_lens) - where_ids_lens[b])
        )
        output_where_array.append(
            output_where_ids[b] + [0] * (max(where_ids_lens) - where_ids_lens[b])
        )
        where_mask_array[b, :where_ids_lens[b]] = 1
    input_where_array = torch.tensor(input_where_array, dtype=torch.long, device=device)
    output_where_array = torch.tensor(output_where_array, dtype=torch.long, device=device)
    where_mask_array = torch.from_numpy(where_mask_array==1).to(device)
    return input_where_array, output_where_array, where_mask_array

