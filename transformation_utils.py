import numpy as np


def get_homography_matrix(src, dst):
    if len(src.shape[1]) == 2:
        src = np.hstack([src, np.ones((len(src), 1))])
    if len(dst.shape[1]) == 2:
        dst = np.hstack([dst, np.ones((len(dst), 1))])

    A = np.zeros((8, 9))
    for i in range(len(src)):
        A[i * 2] = np.array([src[i][0], src[i][1], 1, 0, 0, 0, -dst[i][0] * src[i][0], -dst[i][0] * src[i][1], -dst[i][0]])
        A[i * 2 + 1] = np.array([0, 0, 0, src[i][0], src[i][1], 1, -dst[i][1] * src[i][0], -dst[i][1] * src[i][1], -dst[i][1]])

    _, _, V = np.linalg.svd(A, full_matrices=True)
    h = V[-1]
    return np.reshape(h, (3, 3))
