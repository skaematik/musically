import math
import random

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from Symbol import *
from vec_noise import snoise2


def inv(img, val=255):
    """ Inverts image
        :arg img
            image to invert
        :arg val
            maximum value in image
        :returns inverted image"""
    return 255-img

# deprecaated


def getSymbols(im, merge_overlap=False):
    """finds and draws bounding boxes around detected objects, returns boxes as Symbols

    im: black and white image - black background, white foreground
    merge_overlap: whether to merge overlapping bounding boxes

    returns: list of Symbols, rgb image with bounding boxes drawn
    """
    def boxes2symbols(boxes):
        symbols = []
        for x, y, w, h in boxes:
            symbols.append(Symbol(SymbolType.UNKNOWN, x, y, w, h))
        return symbols

    def draw(im, boxes):
        for x, y, w, h in boxes:
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)

    def box2pts(box):
        x, y, w, h = box
        return (x, y), (x + w, y + h)

    def merge_overlaps(boxes):
        def overlapping(b1, b2):
            l1, r1 = box2pts(b1)
            l2, r2 = box2pts(b2)
            return not (l1[0] > r2[0] or l2[0] > r1[0] or l1[1] > r2[1] or l2[1] > r1[1])

        def next():
            for i in range(len(boxes)):
                for j in range(len(boxes)):
                    if i < j and overlapping(boxes[i], boxes[j]):
                        b1 = boxes.pop(j)
                        b2 = boxes.pop(i)
                        return b1, b2
            return None

        while True:
            pair = next()
            if pair is None:
                break
            b1, b2 = pair
            l1, r1 = box2pts(b1)
            l2, r2 = box2pts(b2)

            x = min(l1[0], l2[0])
            x_ = max(r1[0], r2[0])
            y = min(l1[1], l2[1])
            y_ = max(r1[1], r2[1])
            w, h = x_ - x, y_ - y
            boxes.append((x, y, w, h))

    _, contours, _ = cv2.findContours(
        im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boxes = [cv2.boundingRect(c) for c in contours]

    colour = np.dstack((im, im, im))

    if merge_overlap:
        merge_overlaps(boxes)

    draw(colour, boxes)

    symbols = boxes2symbols(boxes)

    return symbols, colour


def add_noise(image, octaves=20, dark_fac=40, freq=3.0, persistence=0.3, lacunarity=0.9):
    """
    some help from https://stackoverflow.com/questions/14435632/impulse-gaussian-and-salt-and-pepper-noise-with-opencv
    """
    freq = freq * octaves
    mean = np.mean(image)
    var = np.var(image)/2
    row, col = image.shape
    noise = np.zeros(image.shape)
    offx = random.random()
    offy = random.random()
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            noise[y, x] = (int((snoise2(offx + x / freq, offy + y / freq,
                                        octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity) + 1) * dark_fac))
    gauss = np.random.normal(mean, math.sqrt(var), (row, col))
    gauss = gauss.reshape(row, col)
    # noise = noise + gauss
    noise[(image < 50)] = 0
    noise = image.astype(np.float64) - noise
    noise = noise / np.max(noise) * 255
    # noise = noise + gauss
    # noise = noise / np.max(noise) * 255
    noise = noise.astype(np.uint8)
    return noise


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def dirty(im):
    return elastic_transform(add_noise(im), im.shape[1] * 2, im.shape[1] * 0.1)
