import numpy as np
import cv2
import skimage.io as skio
import skimage as sk

import robust as robust
import fusion
import patch_matching
import denoise
import color_transfer
import pdb

PYR_SIZE = 2
OPT_ITERATIONS = 3
PATCH_SIZES = [33, 21, 13, 9]
SUB_SAMPLING_GAPS = [28, 18, 8, 5]

# PYR_SIZE = 3
# OPT_ITERATIONS = 3
# PATCH_SIZES = [21, 13]
# SUB_SAMPLING_GAPS = [18, 8]

"""
Input: Style image (3-D), content image (3-D), optional
weight image (2-D, same size as content image).
Output: Content image transformed to have the style of the style image
"""
def style_transfer(style, content, weight=None):
    if weight is None:
        weight = np.ones(content.shape)

    # Transfer color to content image first  
    s_pyr = style.copy()
    c_pyr = content.copy()
    c_pyr = color_transfer.color_transfer(s_pyr, c_pyr)
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

    X = content_pyr[0] + np.random.normal(scale=50, size=content_pyr[0].shape)
    # Loop over every size
    for l in range(0, PYR_SIZE):
        style_l = style_pyr[l]
        content_l = content_pyr[l]
        weight_l = weight_pyr[l]

        for patch_size, sample_gap in zip(PATCH_SIZES, SUB_SAMPLING_GAPS):
            if patch_size > X.shape[0] or patch_size > X.shape[1] or patch_size > style_l.shape[0] or patch_size > style_l.shape[1]:
                continue
            patch_matcher = patch_matching.PatchMatcher(style_l, patch_size)
            cv2.imwrite("results/init" + str(l) + str(patch_size) + ".jpg", X)
            for i in range(OPT_ITERATIONS):
                print("pyramid level:", l, "patch size:", patch_size, "iteration", i)
                neighborhoods, matches = patch_matcher.find_nearest_neighbors(X, sample_gap)
                #cv2.imwrite("results/matches.jpg", np.vstack(matches[:10]))
                X_tilde = robust.robust_agg(neighborhoods, matches, X, patch_size)
                cv2.imwrite("results/robust" + str(l) + str(patch_size) + ".jpg", X_tilde)
                X_hat = fusion.content_fusion(X_tilde, content_l, weight_l)
                cv2.imwrite("results/fusion" + str(l) + str(patch_size) + ".jpg", X_hat)
    
                X_colored = color_transfer.color_transfer(style_l, X_hat) # should be x_hat
                X = denoise.denoise(X_colored)
                cv2.imwrite("results/output" + str(l) + str(patch_size) + ".jpg", X)
            break
    
        if l + 1 < PYR_SIZE:
            X = cv2.resize(X, (content_pyr[l+1].shape[1], content_pyr[l+1].shape[0]))

            # add noise for next layer, update this std dev
            X = X + np.random.normal(scale=50, size=X.shape) #updat std dev?
    return X

if __name__ == "__main__":
    import datetime
    print("started", datetime.datetime.now())
    style = cv2.imread("images/starry_med.jpg")
    content = cv2.imread("images/white.png") / 2.0
    # weight_raw = cv2.imread("images/cat_small_mask_head.jpg")
    # weight = np.zeros(content.shape)
    # weight[weight_raw > 0] = 1
    weight = None
    X = style_transfer(style, content, weight)
    cv2.imwrite("style_transfer_output_full_boy.png", X)
    print("ended", datetime.datetime.now())




    
