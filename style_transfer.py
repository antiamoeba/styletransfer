import numpy as np
import cv2

PYR_SIZE = 5

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
    
