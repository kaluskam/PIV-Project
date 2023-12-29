import os
import imageio.v3 as iio
import cv2

from pathlib import Path

def load_video(path):
    return iio.imread(path, plugin="pyav")


def load_frame_from_video(path, frame_id):
    return iio.imread(
        path,
        index=frame_id,
        plugin="pyav")


def load_n_frames(filename, out_dir, n_steps):
    video_mat = load_video(filename)  # returns a NumPy array
    video_mat = video_mat[:n_steps, :, :, :]  # subsample

    # Saving the imagaes
    for frame_n in range(video_mat.shape[0]):
        frame_name = f"{out_dir}/frame_{frame_n}.jpg"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        iio.imwrite(frame_name, video_mat[frame_n])
    return video_mat
    

def video_to_frames(filename, out_dir, n_steps):
    vidcap = cv2.VideoCapture(str(filename))
    success, image = vidcap.read()
    count = 0
    while success: 
        success, image = vidcap.read()
        if count % n_steps == 0:
            output_path = out_dir / f"frame_{count//n_steps}.jpg"
            cv2.imwrite(str(output_path), image)     # save frame as JPEG file      
        count += 1
