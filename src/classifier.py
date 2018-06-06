from keras.models import load_model

from Segmenter import Segmenter

import numpy as np
import tensorflow as tf


class Classifier:
    def __init__(self):
        print('loading keras model...')
        self.g = tf.Graph()
        with self.g.as_default():
            print("Loading model")
            self.model = load_model('./resources/model/keras_modelv3.h5')
        print('loaded keras model')
        self.labels = ['barline', 'clef', 'eighth', 'half', 'quarter', 'rest_eighth',
                       'rest_measure', 'rest_quarter', 'tied_eighth', 'timeSig', 'whole']

    def predict_single(self, im):
        symImgs = [np.expand_dims(im, axis=3)]
        y = self.model.predict_classes(np.asarray(
            symImgs), batch_size=len(symImgs), verbose=1)
        return self.labels[y[0]]

    def predict_symbols(self, symbols, use_class_numbers=False):
        symImgs = [np.expand_dims(x.im, axis=3) for x in symbols]
        with self.g.as_default():
            y = self.model.predict_classes(np.asarray(
                symImgs), batch_size=len(symImgs), verbose=1)
        if use_class_numbers:
            return y
        return [self.labels[i] for i in y]

    def predict_probabilities_single(self, im):
        symImgs = [np.expand_dims(im, axis=3)]
        y = self.model.predict_proba(np.asarray(
            symImgs), batch_size=len(symImgs), verbose=1)
        return y

    def predict_probabilities_symbols(self, symbols):
        symImgs = [np.expand_dims(x.im, axis=3) for x in symbols]
        y = self.model.predict_proba(np.asarray(
            symImgs), batch_size=len(symImgs), verbose=1)
        return y
