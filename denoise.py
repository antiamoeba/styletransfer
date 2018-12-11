import numpy as np
import cv2

def denoise(x):
    # used flag=2 because apparently good for stylizations idk
    # apparently a partial implementation of domain transform filtering
    return cv2.edgePreservingFilter(x, flags=2, sigma_s=5, sigma_r=0.05)


if __name__ == "__main__":
    img = cv2.imread("images/cow.jpg")
    denoised = denoise(img)
    cv2.imwrite("denoised_test.png", denoised)
