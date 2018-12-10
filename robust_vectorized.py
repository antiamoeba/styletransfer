import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import pdb

R = 0.8 # for robust aggregation (avoid least squares)
ITERATIONS = 3 #10 #IRLS iterations

def robust_agg(centers, patches, x, patch_size):
    # calculate weights
    num_rows = x.shape[0] 
    num_cols = x.shape[1]
    total_px = num_rows * num_cols * 3
    yw, xw = np.meshgrid(np.arange(patch_size), np.arange(patch_size))
    yw, xw = yw.flatten(), xw.flatten()
    yw3, xw3, cw3 = np.meshgrid(np.arange(patch_size), np.arange(patch_size), np.arange(3))
    yw3, xw3, cw3 = yw3.flatten(), xw3.flatten(), cw3.flatten()
    style = patches.reshape(patches.shape[0], -1)
    patch_num = np.arange(patches.shape[0])
    x_tilde = x
    for iteration in range(ITERATIONS):
        x_nh = x_tilde[centers[0][:, None] + yw, centers[1][:, None]]
        weight = np.linalg.norm(patches.reshape(x_nh.shape)-x_nh, axis=(1,2)) ** (R - 2)
        weight_rt = weight ** 0.5
        index = ((centers[0][:, None] + yw3) * num_cols + (centers[1][:, None] + xw3)) * 3 + cw3
        rows = np.arange(index.size)
        cols = index.flatten()
        vals = np.repeat(weight_rt, patch_size*patch_size*3)
        b = (weight_rt[:, None] * style).flatten()
        A = csr_matrix((vals, (rows, cols)), shape=(len(rows), total_px))

        output = linalg.lsqr(A, b)
        output_data = output[0]
        x_tilde = np.reshape(output_data, (num_rows, num_cols, 3))
    return x_tilde
    
if __name__ == "__main__":
    img = np.random.rand(20, 30, 3)
    centers = np.array([[0, 0, 0, 10, 10, 10],
               [0, 10, 20, 0, 10, 20]])
    patches = np.array([np.random.rand(10, 10, 3) for _ in range(centers.shape[1])])
    patch_size = 10
    agg = robust_agg(centers, patches, img, patch_size)
