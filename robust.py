import numpy as np
import cv2
from scipy.sparse import csr_matrix
from scipy.sparse import linalg

R = 0.8 # for robust aggregation (avoid least squares)
ITERATIONS = 10 #IRLS iterations

def naive_agg(centers, patches, X, patch_size):
    centers = centers.T
    # calculate weights
    num_rows = X.shape[0] 
    num_cols = X.shape[1]
    total_px = num_rows * num_cols * 3

    def get_index(y, x, k):
        return (y * num_cols + x) * 3 + k

    x_tilde = X.copy()
    for i in range(len(centers)):
        center = centers[i]
        x_tilde[center[0]: center[0]+patch_size, center[1]: center[1]+patch_size] = patches[i].copy()

    return x_tilde

def less_robust_agg(centers, patches, X, patch_size):
    centers = centers.T
    # calculate weights
    num_rows = X.shape[0] 
    num_cols = X.shape[1]
    total_px = num_rows * num_cols * 3

    def get_index(y, x, k):
        return (y * num_cols + x) * 3 + k

    x_tilde = X
    rows = []
    cols = []
    vals = []
    b = []
    curr = 0
    for i in range(len(centers)):
        center = centers[i]
        x_nh = x_tilde[center[0]: center[0]+patch_size, center[1]: center[1]+patch_size]
        style_nh = patches[i]

        for i in range(patch_size):
            for j in range(patch_size):
                for k in range(3):
                    index = get_index(center[0] + i, center[1] + j, k)
                    rows.append(curr)
                    cols.append(index)
                    vals.append(1)
                    b.append(style_nh[i, j, k])
                    curr += 1

    b = np.array(b)
    A = csr_matrix((vals, (rows, cols)), shape=(curr, total_px))

    output = linalg.lsqr(A, b)
    output_data = output[0]
    x_tilde = np.reshape(output_data, (num_rows, num_cols, 3))
    return x_tilde
    
def robust_agg(centers, patches, X, patch_size):
    centers = centers.T
    # calculate weights
    num_rows = X.shape[0] 
    num_cols = X.shape[1]
    total_px = num_rows * num_cols * 3

    def get_index(y, x, k):
        return (y * num_cols + x) * 3 + k

    x_tilde = X
    for iteration in range(ITERATIONS):
        rows = []
        cols = []
        vals = []
        b = []
        curr = 0
        for i in range(len(centers)):
            center = centers[i]
            x_nh = x_tilde[center[0]: center[0]+patch_size, center[1]: center[1]+patch_size]
            style_nh = patches[i]

            diff = np.linalg.norm(style_nh - x_nh)
            weight = diff ** (R - 2)
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

# Testing
if __name__ == "__main__":
    img = np.random.rand(20, 30, 3)
    centers = np.array([[0, 0, 0, 10, 10, 10],
               [0, 10, 20, 0, 10, 20]])
    patches = np.array([np.random.rand(10, 10, 3) for _ in range(centers.shape[1])])
    patch_size = 10
    agg = robust_agg(centers, patches, img, patch_size)
