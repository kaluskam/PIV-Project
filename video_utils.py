import os
import imageio.v3 as iio

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
