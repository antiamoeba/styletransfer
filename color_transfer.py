import numpy as np
import cv2
import skimage.io as skio
import skimage as sk

# TODO: Make more efficient so we don't have to compute
# the histogram each time
"""
Input: Style image S, Guess Image X

Output: Guess X, with transferred color through histogram matching
source: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
"""
def color_transfer(S3, X3):
    channels_S = cv2.split(S3)
    channels_X = cv2.split(X3)

    output_channels = []
    for S, X in zip(channels_S, channels_X):
        old_shape = X.shape
        X = X.ravel()
        S = S.ravel()
        X_values, bin_idx, X_counts = np.unique(X, return_inverse=True, return_counts=True)
        S_values, S_counts = np.unique(S, return_counts=True)
        X_quantiles = np.cumsum(X_counts).astype(np.float64)
        X_quantiles = X_quantiles / X_quantiles[-1]
        S_quantiles = np.cumsum(S_counts).astype(np.float64)
        S_quantiles = S_quantiles / S_quantiles[-1]
        interp_S_values = np.interp(X_quantiles, S_quantiles, S_values)
        output_channels.append(interp_S_values[bin_idx].reshape(old_shape))
    return cv2.merge(output_channels)


if __name__ == "__main__":
    img1 = sk.img_as_float(skio.imread("images/sunday_afternoon.png"))
    img2 = sk.img_as_float(skio.imread("images/cow.jpg"))

    output_img1 = color_transfer(img1, img2)
    skio.imsave("color_transfer_test.jpg", output_img1)
    skio.imshow(output_img1)
    skio.show()
