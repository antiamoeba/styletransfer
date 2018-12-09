import numpy as np
import cv2
import skimage.io as skio
import skimage as sk

import robust
import fusion
import patch_matching
import denoise
import color_transfer

# PYR_SIZE = 5
# OPT_ITERATIONS = 10
# PATCH_SIZES = [33, 21, 13, 9]
# SUB_SAMPLING_GAPS = [28, 18, 8, 5]
PYR_SIZE = 3
OPT_ITERATIONS = 3
PATCH_SIZES = [21, 13]
SUB_SAMPLING_GAPS = [18, 8]

"""
Input: Style image (3-D), content image (3-D), optional
weight image (2-D, same size as content image).
Output: Content image transformed to have the style of the style image
"""
def style_transfer(style, content, weight=None):
    if not weight:
        weight = np.ones(style.shape)

    s_pyr = style.copy()
    c_pyr = content.copy()
    w_pyr = weight.copy()

    style_pyr = [s_pyr]
    content_pyr = [c_pyr]
    weight_pyr = [w_pyr]

    for i in range(1, PYR_SIZE):
        c_pyr = cv2.pyrDown(c_pyr)
        s_pyr = cv2.pyrDown(s_pyr)
        w_pyr = cv2.pyrDown(w_pyr)

        content_pyr.append(c_pyr)
        style_pyr.append(s_pyr)
        weight_pyr.append(w_pyr)

    content_pyr, style_pyr, weight_pyr = content_pyr[::-1], style_pyr[::-1], weight_pyr[::-1]

    X = content_pyr[0] + np.random.normal(scale=.2, size=content_pyr[0].shape)

    # Loop over every size
    for l in range(0, PYR_SIZE):
        style_l = style_pyr[l]
        content_l = content_pyr[l]
        weight_l = content_pyr[l]

        for patch_size, sample_gap in zip(PATCH_SIZES, SUB_SAMPLING_GAPS):
            patch_matcher = patch_matching.PatchMatcher(style_l, patch_size)
            for i in range(OPT_ITERATIONS):
                print("matching")
                neighborhoods, matches = patch_matcher.find_nearest_neighbors(X, sample_gap)
                print("robust")
                X_tilde = robust.robust_agg(neighborhoods, matches, X, patch_size)
                print("fusion")
                X_hat = fusion.content_fusion(X_tilde, content_l, weight_l)
                print("color")
                X_colored = color_transfer.color_transfer(style_l, X_hat)
                print("denoise")
                X = denoise.denoise(X_colored)

        if l + 1 < PYR_SIZE:
            X = cv2.resize(X, (content_pyr[l+1].shape[1], content_pyr[l+1].shape[0]))

            # add noise for next layer, update this std dev
            X = X + np.random.normal(scale=.1, size=X.shape) #updat std dev?
    return X

if __name__ == "__main__":
    style = sk.img_as_float(cv2.imread("images/starry_tiny.jpg"))
    content = sk.img_as_float(cv2.imread("images/cat_small.jpg"))
    X = style_transfer(style, content)
    cv2.imwrite("style_transfer_output2.png", X)




    
