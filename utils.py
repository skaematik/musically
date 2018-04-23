from collections import Counter
import cv2
import numpy as np


def import_img_greyscale(file_name):
    """Loads image from file as greyscale
        :raises FileNotFoundError
        :returns greyscale image"""
    img = cv2.imread(file_name, 0)
    if img is None:
        raise FileNotFoundError
    return img


def inv(img, val):
    """ Inverts image
        :arg img
            image to invert
        :arg val
            maximum value in image
        :returns inverted image"""
    return 255-img


def calculate_staff_values(img):
    """Calculates the staff_width and staff height
        staffwidth is the thickness of the black lines
        staffheight is the white between black lines
        most common runtime encodingnto calculate values
        :arg img
            binary image of music, makr sure music is in horizontal lines

        :returns staff_width, staff_height"""
    blackruns = Counter()
    whiteruns = Counter()  # key is run length val is count of this run length
    count = 0
    is_white = True
    for x in range(img.shape[1]):
        count = 0
        for y in range(img.shape[0]):
            cur_white = img[y, x] == 255
            if cur_white != is_white:
                if is_white:
                    whiteruns[count] += 1 # record this run
                else:
                    blackruns[count] += 1  # record this run
                count = 0
            is_white = cur_white
            count += 1
    print('blackRun: {}', blackruns.most_common(1))
    print('whiteRun: {}', whiteruns.most_common(1))
    return blackruns.most_common(1), whiteruns.most_common(1)