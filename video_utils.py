import imageio.v3 as iio


def load_video(path):
    return iio.imread(path,  plugin="pyav")


def load_frame_from_video(path, frame_id):
    return iio.imread(
        path,
        index=frame_id,
        plugin="pyav")

