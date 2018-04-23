from collections import Counter

import cv2
import numpy as np

from utils import *


def main():
    img = import_img_greyscale('./sheets/oneliner.png')
    cv2.imshow('pre', img)
    thresh_img = cv2.adaptiveThreshold(inv(img,255),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,0)
    out = inv(cv2.morphologyEx((thresh_img), cv2.MORPH_RECT, np.ones((3, 1), np.uint8)), 255)
    cv2.imshow('post', out)
    runFinder = inv(thresh_img,255)
    cv2.imshow('run found on', runFinder)
    staff_width, staff_height = calculate_staff_values(runFinder)
    print('width: {} height: {}'.format(staff_width, staff_height))
    out = inv(cv2.morphologyEx(inv(out,255), cv2.MORPH_DILATE, np.ones((2, 2), np.uint8)), 255)
    out = inv(cv2.morphologyEx(inv(out,255), cv2.MORPH_DILATE, np.ones((2, 2), np.uint8)), 255)
    out = inv(cv2.morphologyEx(inv(out,255), cv2.MORPH_DILATE, np.ones((2, 2), np.uint8)), 255)
    out = inv(cv2.morphologyEx(inv(out,255), cv2.MORPH_DILATE, np.ones((2, 2), np.uint8)), 255)
    out = inv(cv2.morphologyEx(inv(out, 255), cv2.MORPH_ERODE, np.ones((2, 2), np.uint8)), 255)
    out = inv(cv2.morphologyEx(inv(out, 255), cv2.MORPH_ERODE, np.ones((2, 2), np.uint8)), 255)
    out = inv(cv2.morphologyEx(inv(out, 255), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), 255)
    # out = inv(cv2.morphologyEx(inv(out, 255), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), 255)
    # out = inv(cv2.morphologyEx(inv(out, 255), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), 255)
    cv2.imshow('post2', out)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()



