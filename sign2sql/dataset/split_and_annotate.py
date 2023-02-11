import os
import sys
sys.path.append('../')
sys.path.append('../../')
from annotate_ws import annotate_example_ws
import ujson as json
from corenlp import CoreNLPClient
from tqdm import tqdm
import copy
from wikisql.lib.common import count_lines, detokenize
from wikisql.lib.query import Query
import numpy as np

VIDEO_PATH = '/mnt/gold/zsj/data/sign2sql/dataset'
DATA_PATH = '/mnt/gold/zsj/data/sign2sql/dataset'
TABLE_PATH = '/mnt/gold/zsj/data/sign2sql/dataset/all.tables.jsonl'
IN_JSONS = [os.path.join(DATA_PATH, 'length%d.jsonl'%i) 
            for i in range(3, 6+1)]
IN_VIDEO_ROOTS = [os.path.join(VIDEO_PATH, 'length%d_preprocessed'%i) 
                for i in range(3, 6+1)]

def load_tables():
    tables = {}
    with open(TABLE_PATH) as ft:
        print('loading tables')
        # ws: Construct table dict with table_id as a key.
        for line in tqdm(ft, total=count_lines(TABLE_PATH)):
            d = json.loads(line)
            tables[d['id']] = d
    return tables

def annotate(json_file_path, tables, video_root):
    fout = json_file_path[:-6]+'_tok.jsonl'
    print('annotating {}'.format(json_file_path))
    with open(json_file_path) as fs, open(fout, 'wt') as fo:
        print('loading examples')
        n_written = 0
        cnt = -1
        for line in tqdm(fs, total=count_lines(json_file_path)):
            cnt += 1
            d = json.loads(line)
            # a = annotate_example(d, tables[d['table_id']])
            a = annotate_example_ws(d, tables[d['table_id']])
            # NEW!! annotate video path
            a['video_path'] = os.path.join(video_root, '%d.npy'%cnt)
            if os.path.exists(a['video_path']):
                fo.write(json.dumps(a) + '\n')
                n_written += 1
        print('wrote {} examples'.format(n_written))
    return fout


def split_dataset(json_file_paths, out_root, ratio=0.15):
    all_data = []
    for jsonf in json_file_paths:
        with open(jsonf) as fs:
            for row in fs.readlines():
                row = row.strip()
                if len(row) != 0:
                    all_data.append(row)
    N = len(all_data)
    print('%d rows in total.' % N)
    idx = np.random.permutation(N)
    with open(os.path.join(out_root, 'train_tok.jsonl'), 'w') as f:
        for id in idx[:int(N*(1.0-ratio*2))]:
            f.write(all_data[id] + '\n')
    with open(os.path.join(out_root, 'dev_tok.jsonl'), 'w') as f:
        for id in idx[int(N*(1.0-ratio*2)): int(N*(1.0-ratio))]:
            f.write(all_data[id] + '\n')
    with open(os.path.join(out_root, 'test_tok.jsonl'), 'w') as f:
        for id in idx[int(N*(1.0-ratio)):]:
            f.write(all_data[id] + '\n')


if __name__ == '__main__':
    tables = load_tables()
    out_files = []
    for in_json, vid_root in zip(IN_JSONS, IN_VIDEO_ROOTS):
        out_files.append(annotate(in_json, tables, vid_root))
    split_dataset(out_files, DATA_PATH)
    
