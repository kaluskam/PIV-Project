
import scipy
import sys

import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import cv2
import glob

import video_utils as vu

def parse_config_file(file_path):
    """
    Reading cfg file that is provided 
    Provided by the professor
    """
    config_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Ignore comments
            if line.startswith('#'):
                continue
            # Split the line into tokens
            tokens = line.split()
            # Extract parameter names and values
            param_name = tokens[0]
            param_values = [tokens[1:]]
            # Check if the token already exists in the dictionary
            if param_name in config_dict:
                # Add new values to the existing token
                config_dict[param_name].extend(param_values)
            else:
                # Create a new entry in the dictionary
                config_dict[param_name] = param_values
    return config_dict

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

VIDEO_PATHS = []
FEATURES_PATHS = []

def load_conf_file():
    global VIDEO_PATHS
    global FEATURES_PATHS
    with open("config_2.cfg") as config_file:
        for line in config_file:
            if line.startswith('videos'):
                VIDEO_PATHS = line.strip().split(' ')[1:]
            if line.startswith('keypoints_out'):
                FEATURES_PATHS = line.strip().split(' ')[1:]

def get_calibration(filename):
    calibration_path = Path("Shared/project/Tesla/CalibrationTesla")
    side_calibration_path = calibration_path / filename
    side_calibration_f = open(side_calibration_path, encoding="utf-8")
    side_calibration_str = side_calibration_f.read()
    K_side, k_side = parse_camera_matrix(side_calibration_str)
    return K_side, k_side

if __name__ == '__main__':
    if len(sys.argv) == 2:
        file_name, config_name = sys.argv  # TODO: Make paths dependent on config

    load_conf_file()
    """
    print("Subsampling video to obtain frames...")
    for video_path in VIDEO_PATHS:
        _, video_name = os.path.split(video_path)
        video_path = Path(video_path)
        video_path_output = Path(f"output/{video_name}/frames")
        FEATURES_PATHS.append(video_path_output)
        os.makedirs(video_path_output, exist_ok=True)
        video_mat =  vu.video_to_frames(video_path, video_path_output, 60)
    print("Saved subsampled frames.")
    """


    K_back, k_back = get_calibration("BackCamera_calibration.txt")
    K_left, k_left = get_calibration("LeftCamera_calibration.txt")
    K_right, k_right = get_calibration("RightCamera_calibration.txt")

    back_images = sorted(glob.glob("Shared/project/Tesla/TeslaVC_carreira/undistorted_images/2023-07-23_11-36-50-back/*.jpg"))
    side_images_left = sorted(glob.glob("Shared/project/Tesla/TeslaVC_carreira/undistorted_images/2023-07-23_11-36-50-left_repeater/*.jpg"))
    side_images_right = sorted(glob.glob("Shared/project/Tesla/TeslaVC_carreira/undistorted_images/2023-07-23_11-36-50-right_repeater/*.jpg"))
    
    for i, (back_i, left_i, right_i) in tqdm(enumerate(zip(back_images, side_images_left, side_images_right))):
        if i > 0:
            break
            
        print(f"back_path: {back_i}")
        print(f"right_patch: {right_i}")

        image1 = cv2.imread(back_i)
        image2 = cv2.imread(left_i)
        image3 = cv2.imread(right_i)

        # Detect keypoints and compute descriptors
        sift = cv2.SIFT_create()
        keypoints_back, descriptors_1 = sift.detectAndCompute(image1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)
        keypoints_3, descriptors_3 = sift.detectAndCompute(image3, None)

        # Match features
        matcher = cv2.BFMatcher()
        matches_left_back = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
        matches_right_back = matcher.knnMatch(descriptors_1, descriptors_3, k=2)

        homographies = []
        
        for image, matches, keypoints in ((image3, matches_right_back, keypoints_3),):
            # Apply ratio test to find good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

            # Extract location of good matches
            points_src = np.zeros((len(good_matches), 2), dtype=np.float32)
            points_dest = np.zeros((len(good_matches), 2), dtype=np.float32)

            for i, match in enumerate(good_matches):
                points_src[i, :] = keypoints_back[match.queryIdx].pt
                points_dest[i, :] = keypoints[match.trainIdx].pt

            # Find homography
            homography, mask = cv2.findHomography(points_src, points_dest, cv2.RANSAC, ransacReprojThreshold=1)

            h_norm = np.linalg.inv(K_back) @ homography  # from homography = K * [rot, trans]
            r_1, r_2, translation = [h_norm[:, column] for column in range(3)]  # Columns from h_norm
            r_3 = np.cross(r_1, r_2)  # Third rotation axis always normal to other two 
            rotation = np.array([r_1, r_2, r_3]) 
            rotation = scipy.linalg.orth(rotation)

            # homographies.append([homography, rotation, translation])

            K_back = np.array(K_back)
            E = K_back.T * homography * K_back
            _, R, t, _ = cv2.recoverPose(E, points_src, points_dest, K_back)
            # points_3D = cv2.triangulatePoints(projMatr1, projMatr2, points_src, points_dest)
            print(f"Rotation: {R}")
            print(f"Translation: {t}")
            homographies.append([homography, R, t])

            # Display the result
            
            height, width, channels = image.shape
            warped_image = cv2.warpPerspective(image1, homography, (width, height))

            image1_kp = cv2.drawKeypoints(image1, keypoints_back, 0, (0, 0, 255))
            image_dest_kp = cv2.drawKeypoints(image, keypoints, 0, (0, 0, 255))
            cv2.imshow('Original Image', image1_kp)
            cv2.imshow('Destination Image', image_dest_kp)
            cv2.imshow('Warped Image', warped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # cv2.imwrite(f"output/back_{i}.jpg", image1_kp)
            # cv2.imwrite(f"output/left_{i}.jpg", image_dest_kp)
            # cv2.imwrite(f"output/warped_{i}.jpg", warped_image)

            

    # Visualization
    # Create a new figure for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw the backcamera, as the origin, since the homographies are relative to it
    i = np.array([1, 0, 0])  # unit vectors along axes
    j = np.array([0, 1, 0])
    k = np.array([0, 0, 1])

    # Plotting the base camera
    back_camera = np.array([0, 0, 0])
    ax.scatter(*back_camera, color='r', s=100, label='Camera Position')
    ax.quiver(*back_camera, *(i), color='r', arrow_length_ratio=0.3)
    ax.quiver(*back_camera, *(j), color='g', arrow_length_ratio=0.3)
    ax.quiver(*back_camera, *(k), color='b', arrow_length_ratio=0.3)

    for index, (homography, rotation, translation) in enumerate(homographies):
        # Plot the basis vectors
        camera_position = back_camera + np.array([translation[0], translation[1], 0])

        # Plot the camera position
        ax.scatter(*camera_position, s=100, label=f"Camera Position {index}")

        # Plot the camera orientation as an arrow
        ax.quiver(*camera_position, *(rotation @ i), color='r', arrow_length_ratio=0.3)
        ax.quiver(*camera_position, *(rotation @ j), color='g', arrow_length_ratio=0.3)
        ax.quiver(*camera_position, *(rotation @ k), color='b', arrow_length_ratio=0.3)

    # Setting plot limits for better visualization
    field_length = 3
    ax.set_xlim([-field_length, field_length])
    ax.set_ylim([-field_length, field_length])
    ax.set_zlim([-field_length, field_length])

    # Labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Camera Position and Orientation')

    # Show the plot
    plt.legend()
    plt.show()

