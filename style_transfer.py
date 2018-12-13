import numpy as np
import cv2

import robust 
import fusion
import patch_matching
import denoise
import color_transfer

PYR_SIZE = 2
OPT_ITERATIONS = 3
PATCH_SIZES = [33, 21, 13, 9]
SUB_SAMPLING_GAPS = [28, 18, 8, 5]

"""
Input: Style image (3-D), content image (3-D), optional
weight image (weight_raw) (2-D, same size as content image),
weight amount (weight_value) to weigh the content image
during fusion, init_color for if we should color transfer
the initial image. Set to False for artistic generation.

Output: Content image transformed to have the style of the
style image
"""
def style_transfer(style, content, weight_value, weight_raw=None, init_color=True):
    if weight_raw is None:
        weight = np.ones(content.shape) * weight_value
    else:
        weight = np.zeros(content.shape)
        weight[weight_raw > 0] = weight_value

    s_pyr = style.copy()
    c_pyr = content.copy()
    if init_color:
        c_pyr = color_transfer.color_transfer(s_pyr, c_pyr)
    w_pyr = weight.copy()

    noise_dev = np.std(c_pyr) * 4

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
    X = content_pyr[0] + np.random.normal(scale=noise_dev, size=content_pyr[0].shape)
    for l in range(0, PYR_SIZE):
        style_l = style_pyr[l]
        content_l = content_pyr[l]
        weight_l = weight_pyr[l]

        for patch_size, sample_gap in zip(PATCH_SIZES, SUB_SAMPLING_GAPS):
            if patch_size > X.shape[0] or patch_size > X.shape[1] or patch_size > style_l.shape[0] or patch_size > style_l.shape[1]:
                continue
            patch_matcher = patch_matching.PatchMatcher(style_l, patch_size)
            cv2.imwrite("results/init_L%i_p%i.jpg" % (l, patch_size), X)
            for i in range(OPT_ITERATIONS):
                print("pyramid level: %i, patch size: %i, iteration: %i" % (l, patch_size, i))
                neighborhoods, matches = patch_matcher.find_nearest_neighbors(
                    X + np.random.normal(scale=noise_dev/4, size=X.shape), sample_gap)
                X_tilde = robust.less_robust_agg(neighborhoods, matches, X, patch_size)
                cv2.imwrite("results/robust_L%i_p%i_i%i.jpg" % (l, patch_size, i), X_tilde)
                X_hat = fusion.content_fusion(X_tilde, content_l, weight_l)
                cv2.imwrite("results/fusion_L%i_p%i_i%i.jpg" % (l, patch_size, i), X_hat)
                X_colored = color_transfer.color_transfer(style_l, X_hat)
                X = X_colored #denoise.denoise(X_colored)
                cv2.imwrite("results/output_L%i_p%i_i%i.jpg" % (l, patch_size, i), X)
        if l + 1 < PYR_SIZE:
            X = cv2.resize(X, (content_pyr[l+1].shape[1], content_pyr[l+1].shape[0]))
            X = X + np.random.normal(scale=noise_dev, size=X.shape)
    return X

# Example usage
if __name__ == "__main__":
    import datetime
    print("started", datetime.datetime.now())
    content = cv2.imread("images/content/bear.jpg")
    style = cv2.imread("images/style/women.jpg")
    weight_raw = None #cv2.imread("images/content/cat_small_mask.png")
    X = style_transfer(style, content, .3, weight_raw=weight_raw, init_color=True)
    cv2.imwrite("results/style_transfer_output_full_bear.png", X)
    print("ended", datetime.datetime.now())




    
