
import scipy.io
import sys

import numpy as np
import pandas as pd
import random

from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

import cv2
from pprint import pprint


def parse_camera_matrix(matrix_str):
    lines = matrix_str.strip().split('\n')
    K = [[0, 0, 0], [0, 0, 0], [0, 0, 1]]  # Intrinsic matrix, assuming [0,0,1] for the last row
    k = []  # Radial distortion coefficients

    for line in lines:
        if line.startswith('#'):
            continue

        values = [float(v) for v in line.split()]
        if len(values) == 4:
            K[0][0] = values[0]  # K(1,1)
            K[1][1] = values[1]  # K(2,2)
            K[0][2] = values[2]  # K(1,3)
            K[1][2] = values[3]  # K(2,3)
        elif len(values) == 3:
            k = values

    return K, k

if __name__ == '__main__':
    file_name, config_name = sys.argv  # TODO: Make paths dependent on config

    calibration_path = Path("Shared/project/Tesla/CalibrationTesla")
    back_calibration_path = calibration_path / "BackCamera_calibration.txt"
    back_calibration_f = open(back_calibration_path, encoding="utf-8")
    back_calibration_str = back_calibration_f.read()

    K_back, k_back = parse_camera_matrix(back_calibration_str)

    transforms_path = Path("output/transforms.mat")
    transforms = scipy.io.loadmat(transforms_path)['transforms']

    _, num_transforms = transforms.shape
    for t_i in tqdm(range(num_transforms)):
        transform = transforms[:, t_i]
        src, dest = transform[:2]
        homography_flat = transform[2:]
        print(f"Movement between frame {src} and frame {dest}")

        homography = homography_flat.reshape((3, 3))
        h_norm = np.linalg.inv(K_back) @ homography  # from homography = K * [rot, trans]
        r_1, r_2, translation = [h_norm[:, column] for column in range(3)]  # Columns from h_norm
        r_3 = np.cross(r_1, r_2)  # Third rotation axis always normal to other two 
        rotation = np.array([r_1, r_2, r_3])

        print(f"Rotation: {rotation}")
        print(f"Tranlation: {translation}")
