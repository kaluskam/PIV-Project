import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import pandas as pd
import open3d as o3d

from pathlib import Path
from PIL import Image

import properties as p
import video_utils as vu
import transformation_utils as tu


if __name__ == "__main__":
    features_path = Path('./output/features.mat')
    mat = scipy.io.loadmat(str(features_path))
    features = mat['features']  # (1, num_frames)
    print(features.shape)
    print(features[0, 2].shape)


    print('END')

