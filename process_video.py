import scipy.io
import numpy as np
import cv2 as cv
import os
import shutil

from tqdm import tqdm
from pathlib import Path

import properties as p
import video_utils as vu

VIDEO_PATHS = []
FEATURE_PATHS = []


def load_video_paths():
    global VIDEO_PATHS
    global FEATURE_PATHS
    config_file = open(p.CONFIG_FILE_PATH)
    for line in config_file:

        if line.startswith('videos'):
            VIDEO_PATHS = line.strip().split(' ')[1:]
        if line.startswith('keypoints_out'):
            FEATURE_PATHS = line.strip().split(' ')[1:]


if __name__ == "__main__":
    print("Running process_video.py ...")
    load_video_paths()
    videos = []
    videos_frames_paths = []
    output_dir = 'output'

    print("Clearing output directory")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    #
    # print("Subsampling video to obtain frames...")
    # for video_path in VIDEO_PATHS:
    #     _, video_name = os.path.split(video_path)
    #     video_path = Path(video_path)
    #     videos.append(vu.load_n_frames(str(video_path), os.path.join(output_dir, video_name, 'frames'), 30))
    # print("Saved subsampled frames.")
    #
    # print("Creating features.mat ...")
    # for j, video in enumerate(videos):
    #     print(f"Creating features for video: {VIDEO_PATHS[j]}")
    #
    #     features = []
    #     _, video_name = os.path.split(VIDEO_PATHS[j])
    #     for frame in tqdm(video):
    #         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #         sift = cv.SIFT_create()
    #         keypoints, descriptors = sift.detectAndCompute(gray, None)
    #
    #         frame_entries = []
    #         for i in range(len(keypoints)):
    #             x, y = keypoints[i].pt
    #             descriptor = descriptors[i]
    #             frame_entry = np.insert(descriptor, 0, (x, y))
    #             frame_entries.append(frame_entry)
    #
    #         data = np.array(
    #             frame_entries)  # Expects shape of ((x,y,descriptor), n_features), not other way around, transpose??
    #         features.append(data)
    #
    #         # Saving as matlab files
    #         features_mat = np.array(features, dtype='object')
    #         for i, el in enumerate(features_mat):
    #             features_mat[i] = el  # el.T
    #         scipy.io.savemat(FEATURE_PATHS[j], {'features': features_mat})
    # print("END")
