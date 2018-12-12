import numpy as np
import cv2
import scipy.sparse as sparse
from scipy.sparse import linalg

def content_fusion(x_tilde, content_img, weights=None):
    assert(x_tilde.shape[0] == content_img.shape[0])
    assert(x_tilde.shape[1] == content_img.shape[1])

    total_px = x_tilde.shape[0] * x_tilde.shape[1] * 3

    if weights is None:
        weights = np.ones(total_px)
    else:
        weights = weights.flatten()

    left_side = 1/(1 + weights)
    right_side = x_tilde.flatten() + np.multiply(weights, content_img.flatten())

    x_hat = np.multiply(left_side, right_side)

    return np.reshape(x_hat, (x_tilde.shape[0], x_tilde.shape[1], 3))

# Testing
if __name__ == "__main__":
    img1 = cv2.imread("images/cow.jpg")
    img2 = cv2.imread("images/mars.jpg")

    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # resize img1 to img2
    output_img = content_fusion(img1, img2)

    # this should just be an average thingy
    cv2.imwrite("color_fusion_test.png", output_img)
