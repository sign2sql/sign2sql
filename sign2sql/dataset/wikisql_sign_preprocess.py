# preprocess sign videos
import os
import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import cv2


# WIKISQL_VIDEO_PATH = '/home/huangwencan/data/sign2sql/'
# VIDEO_ROOT_LIST = [os.path.join(WIKISQL_VIDEO_PATH, 'length%d'%i)  for i in range(3, 6+1)]

WIKISQL_VIDEO_PATH = '/mnt/gold/zsj/poseformat/'
TARGET_DIR = "/mnt/gold/zsj/data/sign2sql/dataset/"
VIDEO_ROOT_LIST = ['length%d'%i  for i in range(3, 6+1)]


def handle_frame(frame):
    x = 255 - frame  # reverse color
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilate = cv2.dilate(x, kernel, 10)  # dilation, #iteration=10
    x2 = cv2.resize(dilate, (144, 144))  # resize to 144, can be 108
    x2 = np.mean(x2/255, axis=-1)  # to gray
    return x2  # (144, 144)


def handle_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    vid_array = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        x = handle_frame(frame)
        vid_array.append(x)
    vid_array = np.asarray(vid_array)
    return vid_array  # (N, 144, 144)


def check(video_root):
    save_root = video_root+'_preprocessed'
    cnt = 0
    for vid_file in os.listdir(video_root):
        save_path = os.path.join(save_root, vid_file[:-4])
        if not os.path.exists(save_path+'.npy'):
            print('Not Exist: '+save_path+'.npy')
            cnt += 1
            # handle
            vid_array = handle_video(os.path.join(video_root, vid_file))
            np.save(save_path, vid_array)
    print('NEW', cnt)


if __name__ == '__main__':
    # for video_root_name in VIDEO_ROOT_LIST:
    #     video_root = os.path.join(WIKISQL_VIDEO_PATH, video_root_name)
    #     check(video_root)
    video_lengths = []
    cnt = 0
    for video_root_name in VIDEO_ROOT_LIST:
        video_root = os.path.join(WIKISQL_VIDEO_PATH, video_root_name)
        for vid_file in os.listdir(video_root):
            vid_array = handle_video(os.path.join(video_root, vid_file))
            video_lengths.append(vid_array.shape[0])
            # save_root = video_root+'_preprocessed'
            save_root = TARGET_DIR+video_root_name+'_preprocessed'
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            save_path = os.path.join(save_root, vid_file[:-4])
            np.save(save_path, vid_array)
            cnt += 1
            if cnt % 100 == 0:
                print('processed %d videos' % cnt)
    print('video length statistics:')
    print(np.min(video_lengths), np.mean(video_lengths), np.max(video_lengths))

