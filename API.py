# import math
# import os
import os.path as osp
# from collections import Counter

# import cv2
# import numpy as np
# import sys
from Segmenter import Segmenter
from Song import Song
from classifier import Classifier
# from utils import *
# from skimage.feature import peak_local_max
# from skimage.morphology import watershed
# from scipy import ndimage
# import matplotlib.pyplot as plt
# from vec_noise import pnoise2, snoise2
# from keras.models import load_model
from music21 import musicxml, environment, instrument
import base64

# from xml_labeler import Xml_labeler

# DEBUG_SHOW_PDF_FILE = False


# def predict(filepath):
def predict(filepath, classifier):
    """
    Input: filepath to the image file (string)
    Output: MusicXML (string) and midi (base 64 string)
    """

    # classifier = Classifier()

    # print(filepath)

    # dir = osp.dirname(filepath)
    basename = osp.basename(filepath)
    filename, extension = osp.splitext(basename)

    # img = cv2.imread(filepath, 0)
    # top = int(0.1 * img.shape[0])
    # img = cv2.copyMakeBorder(
    # img, top, top, top, top, cv2.BORDER_CONSTANT, None, 255)

    # cv2.imwrite('./sheets/tmp/{}_white.png'.format(basename), img)
    # symbols, seg = Segmenter.symbols_from_File(
    #     './sheets/tmp/{}_white.png'.format(basename), use_cache=True)
    symbols, seg = Segmenter.symbols_from_File(filepath, use_cache=True)
    y = classifier.predict_symbols(symbols, use_class_numbers=True)
    # yy = classifier.predict_probabilities_symbols(symbols)
    for i in range(len(y)):
        symbols[i].work_out_type(y[i])
        # cv2.imwrite('./tests/tmp/classifer/{}_{}_{}.png'.format(
        #     basename, symbols[i].get_name(), i), symbols[i].im)
    # cv2.imwrite(
    #     './tests/tmp/classifer/{}__boxed.png'.format(basename), seg.getSymbols(True))

    song = Song()
    song.add_symbols(symbols)
    song.parse_symbols()

    # if DEBUG_SHOW_PDF_FILE:
    #     environment.set(
    #         'musescoreDirectPNGPath',
    #         'C:\\Program Files (x86)\\MuseScore 2\\bin\\MuseScore.exe')
    #     song.stream.show('musicxml.pdf')

    # song.stream.show()

    # for el in song.stream.recurse():
    #     if 'Instrument' in el.classes:  # or 'Piano'
    #         el.activeSite.replace(el, instrument.Piano())

    # for p in song.stream.Score().parts:
    #     p.insert(0, instrument.Piano())

    out = musicxml.m21ToXml.GeneralObjectExporter(song.stream).parse()
    outStr = out.decode('utf-8')

    song.stream.write('midi', osp.join('mid', filename + '.mid'))
    file = open(osp.join('mid', filename+'.mid'), 'rb').read()
    b64 = base64.b64encode(file)

    return outStr.replace('\n', ''), b64


if __name__ == "__main__":
    classifier = Classifier()
    predict("./sheets/test_piece2-1.png", classifier)
