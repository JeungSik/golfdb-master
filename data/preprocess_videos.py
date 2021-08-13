import pandas as pd
import os
import cv2
from multiprocessing import Pool
import numpy as np

df = pd.read_pickle('golfDB.pkl')
youtube_video_dir = './videos'


def preprocess_videos(anno_id, dim=480):
    """
    Extracts relevant frames from youtube videos
    """
    a = anno_id
    # a = df.loc[df['id'] == anno_id]
    bbox = a['bbox']
    # bbox = a['bbox'][0]
    events = a['events']
    # events = a['events'][0]

    path = 'videos_{}/'.format(dim)

    if not os.path.isfile(os.path.join(path, "{}.mp4".format(anno_id['id']))):
        print('Processing annotation id {}'.format(anno_id['id']))
        cap = cv2.VideoCapture(os.path.join(youtube_video_dir, '{}.mp4'.format(a['youtube_id'])))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(os.path.join(path, "{}.mp4".format(anno_id['id'])),
                              fourcc, cap.get(cv2.CAP_PROP_FPS), (dim, dim))
        x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[0])
        y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[1])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * bbox[2])
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * bbox[3])
        count = 0
        success, image = cap.read()
        while success:
            count += 1
            if count >= events[0] and count <= events[-1]:
                    crop_img = image[y:y + h, x:x + w]
                    crop_size = crop_img.shape[:2]
                    ratio = dim / max(crop_size)
                    new_size = tuple([int(x*ratio) for x in crop_size])
                    resized = cv2.resize(crop_img, (new_size[1], new_size[0]))
                    delta_w = dim - new_size[1]
                    delta_h = dim - new_size[0]
                    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                    left, right = delta_w // 2, delta_w - (delta_w // 2)
                    b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                               value=[0.406*255, 0.456*255, 0.485*255])  # ImageNet means (BGR)
                    out.write(b_img)
            if count > events[-1]:
                break
            success, image = cap.read()
    else:
        print('Annotation id {} already completed for size {}'.format(anno_id['id'], dim))


if __name__ == '__main__':
    path = 'videos_{}/'.format(480)
    if not os.path.exists(path):
        os.mkdir(path)

    # preprocess_videos(df.id[1])

    for row_index, value in df.iterrows():
        print('Preprocess {} : {}.mp4 .......'.format(value['id'], value['youtube_id']))

        # Start preprocess videos
        preprocess_videos(df.loc[value['id']])

    # p = Pool(6)
    # p.map(preprocess_videos, df.id)
