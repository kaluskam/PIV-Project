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

def get_point_entry(x, y, u, v):
    # See: https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
    return np.array([
        [-x, -y, -1, 0, 0, 0, u * x, u * y, u], 
        [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    ])


if __name__ == "__main__":
    features_path = Path('./output/features.mat')
    mat = scipy.io.loadmat(str(features_path))
    features = mat['features']  # (1, num_frames)

    base_path = Path('./Shared/project/Tesla/Tecnico_originals/images_11_36_50/back/')
    src_i = 1
    dest_i = src_i + 1

    frame_src = features[0, src_i]
    frame_dest = features[0, dest_i]

    src_path = base_path / f"back_undistorted_{src_i:04}.jpg"
    dest_path = base_path / f"back_undistorted_{dest_i:04}.jpg"

    # Find the best match for each descriptor in the first set
    print("Feature matching...")

    matches = []
    distances = cdist(frame_src.T, frame_dest.T, 'euclidean')
    for i, distance in enumerate(distances):
        min_index = np.argmin(distance)
        matches.append((i, min_index))

    print(f"First 3 Matches:")
    for src, dest in random.sample(matches, 3):
        x, y = frame_src[:2, src]
        u, v = frame_dest[:2, dest]
        print(f"{(x, y)} -> {(u, v)}")
    print()

    # RANSAC
    best_model = None
    threshold = 30
    inlier_max = 0
    n_sample = 4
    for i in tqdm(range(100)):
        # print(f"Sampling {n_sample} points to perform optimization wrt to finding the homography...")
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
        inlier_count = 0

        # Check if model better
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
    homography = best_model

    src_img = np.array(Image.open(src_path))
    dest_img = Image.open(dest_path)

    print("Writing the image to the disk...")
    output_img = np.zeros_like(src_img)
    rows, cols = src_img.shape[:2]
    for x in range(rows):
        for y in range(cols):
            input_v = np.array([x, y, 1])
            output_v = homography @ input_v
            output_scaled = output_v / output_v[2]

            x_new, y_new, _ = output_scaled
            if 0 <= x_new < rows and 0 <= y_new < cols:
                output_img[int(x_new), int(y_new)] = src_img[x, y]

    warped_img = Image.fromarray(output_img)   
    warped_img.save("./output/_warped_img.jpg")

    src_img = Image.open(src_path)
    src_img.save("./output/_src_img.jpg") 
    dest_img.save("./output/_dest_img.jpg")  

    print('END')