from argparse import Namespace
import os

from config import *
from Detection.YOLOv3.detector import YOLOv3
from AlgorithmAnalysis import analysis

video_files = []
full_path_video_files = []
files_list = os.listdir(FOLDER_WITH_VIDEO_FILES)
for file in files_list:
    if os.path.isfile(os.path.join(FOLDER_WITH_VIDEO_FILES, file)) and \
            os.path.splitext(file)[1] in ['.mp4', '.avi', '.mkv', '.mov']:
        video_files.append(file)
        full_path_video_files.append(os.path.join(FOLDER_WITH_VIDEO_FILES, file))

for video in video_files:
    args = Namespace(input=video, outdir=SAVE_PATH_OF_VIDEO_FILES, cuda=cuda, no_show=False)
    YOLOv3.run(args)
    print("Video:", video, 'processed')

analysis(video_files, FOLDER_WITH_VIDEO_FILES, SAVE_PATH_OF_VIDEO_FILES)
