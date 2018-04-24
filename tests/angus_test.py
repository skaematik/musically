from collections import Counter

import cv2
import numpy as np

from utils import *
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import matplotlib.pyplot as plt

def main():
    img = import_img_greyscale('./sheets/ode-to-joy.png')
    cv2.imshow('pre', img)
    thresh_img = inv(cv2.adaptiveThreshold(inv(img,255),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,0))
    out = thresh_img # inv(cv2.morphologyEx((thresh_img), cv2.MORPH_RECT, np.ones((3, 1), np.uint8)), 255)
    cv2.imshow('post', out)
    find_staffs(out,show_plots=True)
    # D = ndimage.distance_transform_edt((out))
    # plt.imshow(D, interpolation='nearest')
    # plt.show()
    # cv2.imshow('Dist', (D).astype(np.uint8))
    # cv2.imshow('Dist on', (out))
    # runFinder = inv(thresh_img,255)
    # cv2.imshow('run found on', runFinder)
    # staff_width, staff_height = calculate_staff_values(runFinder)
    # print('width: {} height: {}'.format(staff_width, staff_height))
    # out = inv(cv2.morphologyEx(inv(out,255), cv2.MORPH_DILATE, np.ones((2, 2), np.uint8)), 255)
    # out = inv(cv2.morphologyEx(inv(out,255), cv2.MORPH_DILATE, np.ones((2, 2), np.uint8)), 255)
    # out = inv(cv2.morphologyEx(inv(out,255), cv2.MORPH_DILATE, np.ones((2, 2), np.uint8)), 255)
    # out = inv(cv2.morphologyEx(inv(out,255), cv2.MORPH_DILATE, np.ones((2, 2), np.uint8)), 255)
    # out = inv(cv2.morphologyEx(inv(out, 255), cv2.MORPH_ERODE, np.ones((2, 2), np.uint8)), 255)
    # out = inv(cv2.morphologyEx(inv(out, 255), cv2.MORPH_ERODE, np.ones((2, 2), np.uint8)), 255)
    # out = inv(cv2.morphologyEx(inv(out, 255), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), 255)
    # # out = inv(cv2.morphologyEx(inv(out, 255), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), 255)
    # # out = inv(cv2.morphologyEx(inv(out, 255), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), 255)
    # cv2.imshow('post2', out)

if __name__ == "__main__":
    main()



