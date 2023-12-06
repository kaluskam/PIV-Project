import os.path

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
FEATURES_PATHS = []
VIDEO_PATHS = []
POINTS = []
transforms_out = None
keypoints_out = None

def read_config_file():
    global VIDEO_PATHS
    global FEATURES_PATHS
    global POINTS
    global transforms_out
    global keypoints_out

    config_file = open(p.CONFIG_FILE_PATH)
    for line in config_file:
        if line.startswith('videos'):
            VIDEO_PATHS = line.strip().split(' ')[1:]
        if line.startswith('transforms_out'):
            transforms_out = Path(line.strip().split(' ')[1])
        if line.startswith('keypoints_out'):
            FEATURES_PATHS = line.strip().split(' ')[1:]


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
    for j, features_path in enumerate(FEATURES_PATHS):
        mat = scipy.io.loadmat(str(features_path))
        features = mat['features']  # (1, num_frames)
        _, video_name = os.path.split(VIDEO_PATHS[j])
        base_path = os.path.join('output', video_name, 'frames')

        transformation_matrix = []
        for src_i in range(1, 2):   # features.shape[1] - 1):  #  Get subsequent homographies between images
            for dest_i in range(1, 3):  # features.shape[1] - 1):

                frame_src = features[0, src_i].T
                frame_dest = features[0, dest_i].T

                src_path = os.path.join(base_path, f"frame_{src_i}.jpg")
                dest_path = os.path.join(base_path, f"frame_{dest_i}.jpg")

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
                threshold = 200
                inlier_max = 0
                n_sample = 4
                print(f"Finding best model with RANSAC...")
                for i in tqdm(range(1000)):
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
                    random.shuffle(matches)
                    for src, dest in matches[:100]:
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

        print(f"Performing pixel wise homography on the image between {src_i}:{dest_i}...")
        warped_img = warp_image(np.array(src_img), best_model)

        warped_img.save("./output/_warped_img.jpg")
        src_img.save("./output/_src_img.jpg")
        dest_img.save("./output/_dest_img.jpg")

        transformation_matrix = np.stack(transformation_matrix, axis=1)
        scipy.io.savemat(transforms_out, {'transforms': transformation_matrix})

        print(f"Transformation matrix {transformation_matrix.shape}...")
        print(transformation_matrix)

    print('END')
