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
    

def video_to_frames(filename, out_filename, n_steps):
    vidcap = cv2.VideoCapture(str(filename))
    success, image = vidcap.read()
    count = 0
    while success: 
        success, image = vidcap.read()
        if count % n_steps == 0:
            cv2.imwrite(f"{out_filename}/frame_{count//n_steps}.jpg", image)     # save frame as JPEG file      
        count += 1
