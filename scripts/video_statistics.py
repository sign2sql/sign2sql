import os
import ujson as json
import numpy as np
DATA_PATH = '/home/huangwencan/data/sign2sql/dataset'
splits = ['train', 'dev', 'test']

if __name__ == '__main__':
    for split in splits:
        print(split, 'start!')
        path_sql = os.path.join(DATA_PATH, split+'_tok.jsonl')
        data = []
        video_lens = []
        cnt = 0
        with open(path_sql) as f:
            for idx, line in enumerate(f):
                t1 = json.loads(line.strip())
                if os.path.exists(t1['video_path']):
                    data.append(t1)
                    video = np.load(t1['video_path'])
                    video_lens.append(video.shape[0])
                    cnt +=1
                if cnt % 100 == 0:
                    print(cnt, 'video loaded')
        print(split, ';', len(video_lens), 'videos; length statistics:')
        print(np.min(video_lens), np.mean(video_lens), np.max(video_lens))
        
