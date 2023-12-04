import imageio.v3 as iio
import cv2
import skvideo.io


def load_video(path):
    return iio.imread(path,  plugin="pyav")


def load_frame_from_video(path, frame_id):
    return iio.imread(
        path,
        index=frame_id,
        plugin="pyav")


def load_n_frames(filename, out_filename, n_steps):
    video_mat = skvideo.io.vread(filename)  # returns a NumPy array
    video_mat = video_mat[::n_steps]  # subsample

    # Saving the imagaes
    for frame_n in range(video_mat.shape[0]):
        frame_name = f"{out_filename}/frame_{frame_n}.jpg"
        skvideo.io.vwrite(frame_name, video_mat[frame_n])
    
    return video_mat
