import math
import os
from collections import Counter

import cv2
import numpy as np

from utils import *
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import matplotlib.pyplot as plt

def main():
    for filename in os.listdir('./sheets/'):
        if filename.endswith('.jpg') or filename.endswith('.jepg') or filename.endswith('.png'):
            print(filename)
            img = cv2.imread(os.path.join('./sheets/',filename))
            img_grey = import_img_greyscale(os.path.join('./sheets/',filename))
            out = inv(cv2.adaptiveThreshold(
                inv(img_grey,255),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,0))
            staff_black, staff_white = calculate_staff_values(out)
            seeds, seed_locs = get_staff_seeds(out, img, staff_black, staff_white)
            staffs, _, staffs_highlighted = get_staff_lines_from_seeds(
                seeds, img_grey, staff_black, staff_white, colour_img=cv2.imread(os.path.join('./sheets/',filename)))
            removed_image = remove_staff_lines(out, staffs, staff_black, staff_white)
            removed_image2 = inv(cv2.morphologyEx(
                inv(removed_image), cv2.MORPH_CLOSE, np.ones((int(math.ceil(staff_black/2)), int(math.ceil(staff_black/2))), np.uint8)))
            removed_image2 = inv(cv2.morphologyEx(
                inv(removed_image2), cv2.MORPH_OPEN, np.ones((int(math.ceil(staff_black/2)), int(math.ceil(staff_black/2))), np.uint8)))
            _, boxed = boundingboxes(inv(removed_image2))
            cv2.imwrite('./tests/output/{}_seeds.png'.format(filename[:-4]), seed_locs)
            cv2.imwrite('./tests/output/{}_staffs.png'.format(filename[:-4]), staffs_highlighted)
            cv2.imwrite('./tests/output/{}_removed.png'.format(filename[:-4]), removed_image)
            cv2.imwrite('./tests/output/{}_removed2.png'.format(filename[:-4]), removed_image2)
            cv2.imwrite('./tests/output/{}_boxed.png'.format(filename[:-4]), boxed)

    return

    def nothing(x):
        pass

    cv2.namedWindow('Colorbars')
    cv2.createTrackbar("RECT {}".format(staff_width*2), "Colorbars", 0, 20, nothing)
    cv2.createTrackbar("CLOSE {}".format(staff_height), "Colorbars", 0, 20, nothing)
    cv2.createTrackbar("OPEN {}".format(staff_height), "Colorbars", 0, 20, nothing)

    while True:
        hul = cv2.getTrackbarPos("RECT {}".format(staff_width * 2), "Colorbars")
        huh = cv2.getTrackbarPos("CLOSE {}".format(staff_height), "Colorbars")
        hug = cv2.getTrackbarPos("OPEN {}".format(staff_height), "Colorbars")
        removed = img_edit = imgs[count]
        if hul is not 0:
            removed = inv(cv2.morphologyEx(inv(removed), cv2.MORPH_RECT, np.ones((int(hul), 1), np.uint8)))
        if huh is not 0:
            removed = inv(cv2.morphologyEx(inv(removed), cv2.MORPH_CLOSE, np.ones((int(huh), int(huh)), np.uint8)))
        if hug is not 0:
            removed = inv(cv2.morphologyEx(inv(removed), cv2.MORPH_OPEN, np.ones((int(hug), int(hug)), np.uint8)))
        cv2.imshow("hozizontal", removed)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            count = (count + 1) % len(imgs)
        elif k == 27:
            break
    cv2.destroyAllWindows()






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



