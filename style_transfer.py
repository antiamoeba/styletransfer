import numpy as np
import cv2

import robust
import fusion
import patch_matching
import denoise
import color_transfer

PYR_SIZE = 5
OPT_ITERATIONS = 10
PATCH_SIZES = [33, 21, 13, 9]

if __name__ == "__main__":
    style = cv2.imread("images/sunday_afternoon_small.png")
    content = cv2.imread("images/cow.jpg")
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

    X = content_pyr[0] + np.random.normal(scale=50, size=content_pyr[0].shape)
    # Loop over every size
    for l in range(0, PYR_SIZE):
        style_l = style_pyr[l]
        content_l = content_pyr[l]
        weight_l = content_pyr[l]

        for patch_size in range(PATCH_SIZES):
            matcher = patch_matching.construct_matcher(style_l, patch_size)
            centers = [] # do this by linspace or sth, or maybe get keys from matcher

            for i in range(OPT_ITERATIONS):
                nn = matcher.match(content_l)

                X_tilde = robust.robust_agg(nn, X, patch_size, centers)
                X_hat = fusion.content_fusion(X_tilde, content_l, weight_l)

                X_colored = color_transfer.color_transfer(style_l, X_hat)

                X = denoise.denoise(X_colored)

        if l + 1 < PYR_SIZE:
            X = cv2.resize(X, (style_pyr[l+1].shape[1], style_pyr[l+1].shape[0]))

            # add noise for next layer, update this std dev
            X = X + np.random.normal(scale=30, size=X.shape)

    cv2.imwrite("style_transfer_output.png", X)




    
