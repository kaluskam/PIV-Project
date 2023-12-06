import scipy.io

import numpy as np
import pandas as pd
import random

from pathlib import Path
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

import properties as p
import video_utils as vu
import transformation_utils as tu

# random.seed(0)

VIDEO_PATHS = []
POINTS = []
transforms_out = None
keypoints_out = None

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
    global transforms_out
    global keypoints_out

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
        if line.startswith('transforms_out'):
            transforms_out = line.strip().split(' ')[1]
        if line.startswith('keypoints_out'):
            keypoints_out = line.strip().split(' ')[1]

def warp_image(src_img, homography):
    output_img = np.zeros_like(src_img)
    rows, cols = src_img.shape[:2]
    for x in tqdm(range(rows)):
        for y in range(cols):
            input_v = np.array([x, y, 1])
            output_v = homography @ input_v
            output_scaled = output_v / output_v[2]

            x_new, y_new, _ = output_scaled
            if 0 <= x_new < rows and 0 <= y_new < cols:
                output_img[int(x_new), int(y_new)] = src_img[x, y]

    warped_img = Image.fromarray(output_img)
    return warped_img

def get_point_entry(x, y, u, v):
    # See: https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
    return np.array([
        [-x, -y, -1, 0, 0, 0, u * x, u * y, u], 
        [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    ])

if __name__ == "__main__":
    print("Reading the configuration...")
    read_config_file()

    features_path = Path(keypoints_out)
    mat = scipy.io.loadmat(str(features_path))
    features = mat['features']  # (1, num_frames)

    base_path = Path('./Shared/project/Tesla/Tecnico_originals/images_11_36_50/back/')

    transformation_matrix = []
    for src_i in range(1, 2): #features.shape[1] - 1):  #  Get subsequent homographies between images
        for dest_i in range(1, 3): #features.shape[1] - 1):

            frame_src = features[0, src_i]
            frame_dest = features[0, dest_i]

            src_path = base_path / f"back_undistorted_{src_i:04}.jpg"
            dest_path = base_path / f"back_undistorted_{dest_i:04}.jpg"

            # Find the best match for each descriptor in the first set
            print(f"Feature matching {src_i} to {dest_i}...")

            matches = []
            distances = cdist(frame_src.T, frame_dest.T, 'euclidean')
            for i, distance in enumerate(distances):
                min_index = np.argmin(distance)
                matches.append((i, min_index))

            # RANSAC
            # TODO: Put into config
            best_model = None
            threshold = 300
            inlier_max = 0
            n_sample = 4
            print(f"Finding best model with RANSAC...")
            for i in tqdm(range(100)):
                top_matches = random.sample(matches, n_sample)

                match_matrix_list = []
                for src, dest in top_matches:
                    x, y = frame_src[:2, src]
                    u, v = frame_dest[:2, dest]
                    match_matrix_list.append(get_point_entry(x, y, u, v))
                match_matrix = np.concatenate(match_matrix_list, axis=0)

                U, S, Vh = np.linalg.svd(match_matrix, full_matrices=True)
                solution = Vh[-1]
                homography = np.reshape(solution, (3, 3))

                # Check if model better
                inlier_count = 0
                for src, dest in matches:
                    x, y = frame_src[:2, src]
                    u, v = frame_dest[:2, dest]
                    input_v = np.array([x, y, 1])
                    output_v = homography @ input_v
                    output_scaled = output_v / output_v[2]
                    x_new, y_new, _ = output_scaled

                    distance = np.linalg.norm(np.array([u, v]) - np.array([x_new, y_new]))
                    if distance < threshold:
                        inlier_count += 1

                if inlier_count > inlier_max:
                    best_model = homography
                    inlier_max = inlier_count
            
            print(f"homography: {best_model}")
            transformation_entry = np.reshape(best_model, (9, 1))
            transformation_entry = np.insert(homography, 0, (dest_i, src_i))
            transformation_matrix.append(transformation_entry)

    src_img = Image.open(src_path)
    dest_img = Image.open(dest_path)

    print("Performing pixel wise homography on the image ...")
    warped_img = warp_image(np.array(src_img), best_model)
    
    warped_img.save("./output/_warped_img.jpg")
    src_img.save("./output/_src_img.jpg")
    dest_img.save("./output/_dest_img.jpg")
    
    transformation_matrix = np.stack(transformation_matrix, axis=1)
    print(f"Transformation matrix {transformation_matrix.shape}...")
    print(transformation_matrix)

    print('END')