import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import pandas as pd
import open3d as o3d
import cv2 as cv

from tqdm import tqdm
from pprint import pprint
from pathlib import Path

from PIL import Image

import properties as p
import video_utils as vu
import transformation_utils as tu

VIDEO_PATHS = []
POINTS = []


class MapFramePair:
    def __init__(self, frame_id, pts_in_map, pts_in_frame):
        self.frame_id = frame_id
        self.map = pts_in_map
        self.frame = pts_in_frame

    def calculate_homography(self):
        # TODO
        #H = tu.get_homography_matrix(self.frame, self.map)
        pass

    def __repr__(self):
        return f'\n' \
               f'frameId: {self.frame_id}\n' \
               f'map:\n{self.map}\n' \
               f'frame:\n{self.frame}\n'


def read_config_file():
    global VIDEO_PATHS
    global POINTS
    config_file = open(p.CONFIG_FILE_PATH)
    for line in config_file:
        if line.startswith('videos'):
            VIDEO_PATHS = line.strip().split(' ')[1:]
        if line.startswith('pts_in_map'):
            pts_in_map = np.array(line.strip().split(' ')[2:]).astype(int)
            pts_in_map = np.reshape(pts_in_map, (pts_in_map.shape[0] // 2, 2))
        if line.startswith('pts_in_frame'):
            frame_id = line.strip().split(' ')[1]
            pts_in_frame = np.array(line.strip().split(' ')[2:]).astype(int)
            pts_in_frame = np.reshape(pts_in_frame, (pts_in_frame.shape[0] // 2, 2))
            POINTS.append(MapFramePair(frame_id, pts_in_map, pts_in_frame))


if __name__ == "__main__":
    print("Running process_video.py")
    read_config_file()
    print(POINTS)
    POINTS[0].calculate_homography()

    mat = scipy.io.loadmat('Shared/project/specs/surf_features.mat')

    print("Subsampling video to obtain frames...")
    # video_path = Path("Shared/project/Tesla/TeslaVC_carreiraVIDEOS/2023-07-23_11-36-50-back.mp4")
    # video_mat =  vu.load_n_frames(str(video_path), "./output", 60)
    print("Saved subsampled frames.")
    
    print("Creating features.mat ...")
    frames = Path('./Shared/project/Tesla/Tecnico_originals/images_11_36_50/back/')
    features = []

    for img_path in tqdm(frames.glob('back_undistorted_*.jpg')):
        img = cv.imread(str(img_path))
        sift = cv.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)

        frame_entries = []
        for i in range(len(keypoints)):
            x, y = keypoints[i].pt
            descriptor = descriptors[i]
            frame_entry = np.insert(descriptor, 0, (x, y))
            frame_entries.append(frame_entry)
        
        data = np.array(frame_entries)  # Expects shape of ((x,y,descriptor), n_features), not other way around, transpose??
        features.append(data)

    # Saving as matlab files
    features_mat = np.array(features, dtype='object')
    for i, el in enumerate(features_mat):
        features_mat[i] = el.T
    feature_path = Path('./output') / "features.mat"
    scipy.io.savemat(feature_path, {'features': features_mat})

    print("END")
