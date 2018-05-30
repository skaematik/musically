from keras.models import load_model

from Segmenter import Segmenter

import numpy as np

class Classifier:



    def __init__(self):
        self.model = load_model('./resources/model/keras_modelv4.h5')
        print('forcing first prediction for speed later')
        segmenter = Segmenter.load_segmenter_from_file('auto_gen_large-002_noise.png')
        symbols = segmenter.symbols
        symImgs = [np.expand_dims(x.im, axis=3) for x in symbols[:4]]
        self.model.predict_classes(np.asarray(symImgs), batch_size=len(symImgs), verbose=1)
        self.labels = ['barline', 'clef', 'eighth', 'half', 'quarter', 'rest_eighth', 'rest_measure', 'rest_quarter', 'tied_eighth', 'timeSig', 'whole']

    def predict_single(self, im):
        symImgs = [np.expand_dims(im, axis=3)]
        y = self.model.predict_classes(np.asarray(symImgs), batch_size=len(symImgs), verbose=1)
        return self.labels[y[0]]

    def predict_symbols(self, symbols, use_class_numbers=False):
        symImgs = [np.expand_dims(x.im, axis=3) for x in symbols]
        y = self.model.predict_classes(np.asarray(symImgs), batch_size=len(symImgs), verbose=1)
        if use_class_numbers:
            return y
        return [self.labels[i] for i in y]

    def predict_probabilities_single(self, im):
        symImgs = [np.expand_dims(im, axis=3)]
        y = self.model.predict_proba(np.asarray(symImgs), batch_size=len(symImgs), verbose=1)
        return y

    def predict_probabilities_symbols(self, symbols):
        symImgs = [np.expand_dims(x.im, axis=3) for x in symbols]
        y = self.model.predict_proba(np.asarray(symImgs), batch_size=len(symImgs), verbose=1)
        return y