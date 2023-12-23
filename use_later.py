POINTS = []


class MapFramePair:
    def __init__(self, frame_id, pts_in_map, pts_in_frame):
        self.frame_id = frame_id
        self.map = pts_in_map
        self.frame = pts_in_frame

    def calculate_homography(self):
        # TODO
        # H = tu.get_homography_matrix(self.frame, self.map)
        pass

    def __repr__(self):
        return f'\n' \
               f'frameId: {self.frame_id}\n' \
               f'map:\n{self.map}\n' \
               f'frame:\n{self.frame}\n'


# loading config file
        # if line.startswith('pts_in_map'):
        #     pts_in_map = np.array(line.strip().split(' ')[2:]).astype(int)
        #     pts_in_map = np.reshape(pts_in_map, (pts_in_map.shape[0] // 2, 2))
        #
        # if line.startswith('pts_in_frame'):
        #     frame_id = line.strip().split(' ')[1]
        #     pts_in_frame = np.array(line.strip().split(' ')[2:]).astype(int)
        #     pts_in_frame = np.reshape(pts_in_frame, (pts_in_frame.shape[0] // 2, 2))
        #     POINTS.append(MapFramePair(frame_id, pts_in_map, pts_in_frame))
