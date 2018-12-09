import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse import linalg

R = 0.8 # for robust aggregation (avoid least squares)
ITERATIONS = 3 #10 #IRLS iterations


def robust_agg(nn, x, patch_size):
    centers = nn.keys()
    # calculate weights
    num_rows = x.shape[0]
    num_cols = x.shape[1]
    total_px = num_rows * num_cols * 3

    def get_index(y, x, k):
        return (y * num_cols + x) * 3 + k

    x_tilde = x
    for iteration in range(ITERATIONS):
        print("Iteration", iteration)
        rows = []
        cols = []
        vals = []
        b = []
        curr = 0
        for center in centers:
            x_nh = x_tilde[center[0]: center[0]+patch_size, center[1]: center[1]+patch_size]
            style_nh = nn[center]
            weight = np.linalg.norm(style_nh - x_nh) ** (R - 2)
            weight_rt = weight ** 0.5

            for i in range(patch_size):
                for j in range(patch_size):
                    for k in range(3):
                        index = get_index(center[0] + i, center[1] + j, k)
                        rows.append(curr)
                        cols.append(index)
                        vals.append(weight_rt)
                        b.append(weight_rt * style_nh[i, j, k])
                        curr += 1

        b = np.array(b)
        A = csr_matrix((vals, (rows, cols)), shape=(curr, total_px))

        output = linalg.lsqr(A, b)
        output_data = output[0]
        x_tilde = np.reshape(output_data, (num_rows, num_cols, 3))

    return x_tilde

    
