"""
youtube_dl을 이용한 Youtube 영상 다운로드 스크립트
"""
import os
import youtube_dl
import pandas as pd
import numpy as np

df = pd.read_pickle('golfDB.pkl')
youtube_video_dir = './videos'      # 비디오 다운로드 경로

def download_video_and_subtitle(output_dir, youtube_video_list):

    download_path = os.path.join(output_dir, '%(id)s.%(ext)s')
    total = 0
    success = 0
    fail = 0
    for video_url in youtube_video_list:
        # youtube_dl options, reference: https://github.com/ytdl-org/youtube-dl/blob/master/README.md#readme
        total += 1

        if not os.path.isfile(os.path.join(youtube_video_dir, "{}.mp4".format(video_url))):
            video_url = "https://www.youtube.com/watch?v="+video_url
            ydl_opts = {
                'format': 'best/best',            # 가장 좋은 화질로 선택(134:360p, 135:480p, 136:720p)
                'outtmpl': download_path    # 다운로드 경로 설정
                # 'autonumber_start': 0       # 자동시작번호 설정
            }

            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                    success += 1
            except Exception as e:
                print('error', e)
                fail += 1
        else:
            success += 1
            print('{}.mp4 already downloaded!!!!!'.format(video_url))

    print('Total {} files Complete download! (Success:{}, Fail:{})'.format(total, success, fail))

if __name__ == '__main__':
    youtube_url_list = [  # 유투브에서 다운로드 하려는 영상의 주소 리스트(아래는 Sample Video 리스트)
        "https://www.youtube.com/watch?v=f1BWA5F87Jc",
        "https://www.youtube.com/watch?v=tA1iotgtMyc",
        "https://www.youtube.com/watch?v=wDCKLePrwHA",
        "https://www.youtube.com/watch?v=iPuVhnl8pJU"
    ]

    youtube_id = df['youtube_id'].tolist()

    # Start download videos
    download_video_and_subtitle(youtube_video_dir, youtube_id)
