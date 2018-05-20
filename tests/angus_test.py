import math
import os
from collections import Counter

import cv2
import numpy as np
import sys
from Segmenter import Segmenter
from utils import *
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import matplotlib.pyplot as plt
from vec_noise import pnoise2, snoise2

def main():

    for filename in os.listdir('./sheets/'):
        if filename.endswith('.jpg') or filename.endswith('.jepg') or filename.endswith('.png'):
            print(filename)
            if filename == "file-page1.png" or filename == "file-page2.png" or filename == "file-page3.png" or \
                    filename[:-4].endswith("noise"):
                continue

            # octaves = 20
            # dark_fac = 40
            # freq = 3.0 * octaves
            # persistence = 0.3
            # lacunarity = 0.9
            img = cv2.imread(os.path.join('./sheets/',filename), 0)
            # noise = np.zeros(img.shape)
            # noisey_img = noisy('gauss', img)
            # for y in range(img.shape[0]):
            #     for x in range(img.shape[1]):
            #         noise[y,x] = (int((snoise2(x / freq, y / freq,
            #                                         octaves,
            #                                         persistence=persistence,
            #                                         lacunarity=lacunarity) + 1) * dark_fac))
            # noise[(img < 30)] = 0
            # noise = img.astype(np.float64) - noise
            # noise = noise / np.max(noise) * 255
            # noise = noise.astype(np.uint8)
            # noise = add_noise(img)
            im_merge_t = elastic_transform(img, img.shape[1] * 2, img.shape[1] * 0.1)

            cv2.imwrite('./sheets/{}_noise.png'.format(filename[:-4]),im_merge_t)
            segmenter = Segmenter(os.path.join('./sheets/',filename))
            img = segmenter.remove_staff_lines()
            cv2.imwrite('./tests/output/{}_removed.png'.format(filename[:-4]), img)
            segmenter = Segmenter(os.path.join('./sheets/', '{}_noise.png'.format(filename[:-4])))
            img = segmenter.remove_staff_lines()
            cv2.imwrite('./tests/output/{}_noise_removed.png'.format(filename[:-4]), img)
    return

    def nothing(x):
        pass

    cv2.namedWindow('Colorbars')
    cv2.createTrackbar("oct 32", "Colorbars", 1, 200, nothing)
    cv2.createTrackbar("multi 64", "Colorbars", 1,100, nothing)
    # cv2.createTrackbar("per /10", "Colorbars", 1, 100, nothing)
    # cv2.createTrackbar("lac /10", "Colorbars", 1, 100, nothing)
    cv2.createTrackbar("freq 16", "Colorbars", 1, 50, nothing)
    img = cv2.imread('./sheets/happy-birthday-to-you.png', 0)

    while True:
        hul = cv2.getTrackbarPos("oct 32", "Colorbars")
        huh = cv2.getTrackbarPos("multi 64", "Colorbars")
        hug = cv2.getTrackbarPos("freq 16", "Colorbars")
        # per = cv2.getTrackbarPos("per /10", "Colorbars") /10
        # lac = cv2.getTrackbarPos("lac /10", "Colorbars") /10
        octaves = hul
        dark_fac =huh
        freq = hug * octaves
        noise = np.zeros(img.shape)
        # noisey_img = noisy('gauss', img)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                noise[y, x] = (int((snoise2(x / freq, y / freq, octaves,persistence=0.3, lacunarity=0.9) + 1) * dark_fac))
        noise[(img < 10)] = img[(img < 10)]
        noise = img.astype(np.float64) - noise
        noise = noise / np.max(noise) * 255
        cv2.imshow("hozizontal", noise.astype(np.uint8))
        k = cv2.waitKey(1000) & 0xFF
        if k == 27:
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



# Define function to draw a grid# Defin
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))

