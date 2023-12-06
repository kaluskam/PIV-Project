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
FEATURES_PATHS = []


def load_conf_file():
    global VIDEO_PATHS
    global FEATURES_PATHS
    with open(p.CONFIG_FILE_PATH) as config_file:
        for line in config_file:

            if line.startswith('videos'):
                VIDEO_PATHS = line.strip().split(' ')[1:]
            if line.startswith('keypoints_out'):
                FEATURES_PATHS = line.strip().split(' ')[1:]


if __name__ == "__main__":
    print("Running process_video.py ...")
    load_conf_file()
    videos = []
    videos_frames_paths = []
    output_dir = 'output'

    print("Clearing output directory")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print("Subsampling video to obtain frames...")

    for video_path in VIDEO_PATHS:
        _, video_name = os.path.split(video_path)
        video_path = Path(video_path)
        videos.append(vu.load_n_frames(str(video_path), os.path.join(output_dir, video_name, 'frames'), 10))
    print("Saved subsampled frames.")

    print("Creating features.mat ...")
    for j in range(len(VIDEO_PATHS)):
        print(f"Creating features for video: {VIDEO_PATHS[j]}")
        _, video_name = os.path.split(VIDEO_PATHS[j])
        frames = os.listdir(os.path.join(output_dir, video_name, 'frames'))
        features = []
        for frame in tqdm(frames):
            img = cv.imread(os.path.join(output_dir, video_name, 'frames', frame))
            sift = cv.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(img, None)

            frame_entries = []
            for i in range(len(keypoints)):
                x, y = keypoints[i].pt
                descriptor = descriptors[i]
                frame_entry = np.insert(descriptor, 0, (x, y))
                frame_entries.append(frame_entry)

            data = np.array(
                frame_entries)  # Expects shape of ((x,y,descriptor), n_features), not other way around, transpose??
            features.append(data)

            # Saving as matlab files
            features_mat = np.array(features, dtype='object')
            for i, el in enumerate(features_mat):
                features_mat[i] = el
            scipy.io.savemat(FEATURES_PATHS[j], {'features': features_mat})
    print("END")
