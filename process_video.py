import os
import numpy as np
import matplotlib.pyplot as plt

import properties as p
import video_utils as vu
import transformation_utils as tu

VIDEO_PATHS = []
POINTS = []


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


if __name__ == "__main__":
    print("Running process_video.py")
    read_config_file()
    print(POINTS)
    POINTS[0].calculate_homography()
    frame = vu.load_frame_from_video("data\\Tesla\\TeslaVC_carreiraVIDEOS\\2023-07-23_11-36-50-back.mp4", 50)
    plt.imsave(os.path.join(p.OUTPUT_DIR, "frame_50.jpg"), frame)
