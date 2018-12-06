import numpy as np
import cv2
import scipy.sparse as sparse
from scipy.sparse import linalg

def content_fusion(x_tilde, content_img, weights=None):
    assert(x_tilde.shape[0] == content_img.shape[0])
    assert(x_tilde.shape[1] == content_img.shape[1])

    total_px = x_tilde.shape[0] * x_tilde.shape[1]

    if not weights:
        weights = np.ones(total_px)

    weights += 1
    
    weight_matrix = sparse.diags(weights, 1)

    left_side = linalg.inv(weight_matrix)
    right_side = x_tilde.flatten() + weight_matrix * content_img.flatten()

    x_hat = left_side * right_side

    print(x_hat.shape)

    return np.reshape(x_hat, (x_tilde.shape[0], x_tilde.shape[1]))
