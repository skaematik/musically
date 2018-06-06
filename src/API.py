import os.path as osp
from Segmenter import Segmenter
from Song import Song
from classifier import Classifier
from music21 import musicxml, environment, instrument
import base64


def predict(filepath, classifier):
    """
    Input: filepath to the image file (string)
    Output: MusicXML (string) and midi (base 64 string)
    """

    basename = osp.basename(filepath)
    filename, extension = osp.splitext(basename)

    symbols, seg = Segmenter.symbols_from_File(filepath, use_cache=True)
    y = classifier.predict_symbols(symbols, use_class_numbers=True)

    for i in range(len(y)):
        symbols[i].work_out_type(y[i])

    song = Song()
    song.add_symbols(symbols)
    song.parse_symbols()

    out = musicxml.m21ToXml.GeneralObjectExporter(song.stream).parse()
    outStr = out.decode('utf-8')

    song.stream.write('midi', osp.join('mid', filename + '.mid'))
    file = open(osp.join('mid', filename+'.mid'), 'rb').read()
    b64 = base64.b64encode(file)

    return outStr.replace('\n', ''), b64


if __name__ == "__main__":
    classifier = Classifier()
    predict("./resources/sheets/test_piece2-1.png", classifier)
