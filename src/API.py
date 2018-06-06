import os
import os.path as osp
from Segmenter import Segmenter
from Song import Song
from classifier import Classifier
from music21 import musicxml, environment, instrument, converter
import base64
from pathlib import Path


def predict(filepath, classifier):
    """
    Input: filepath to the image file (string)
    Output: MusicXML (string) and midi (base 64 string)
    """

    basename = osp.basename(filepath)
    filename, extension = osp.splitext(basename)

    cached_file = Path(osp.join("cached_stream", filename + '.pkl'))
    if cached_file.is_file():
        song_stream = converter.thaw(cached_file)
    else:

        symbols, seg = Segmenter.symbols_from_File(filepath, use_cache=True)
        y = classifier.predict_symbols(symbols, use_class_numbers=True)

        for i in range(len(y)):
            symbols[i].work_out_type(y[i])

        song = Song()
        song.add_symbols(symbols)
        song.parse_symbols()

        converter.freeze(song.stream, fmt='pickle', fp=osp.join(os.getcwd(),"cached_stream", filename + ".pkl"))
        song_stream = song.stream

    # get musicxml as a string
    out = musicxml.m21ToXml.GeneralObjectExporter(song_stream).parse()
    outStr = out.decode('utf-8')

    # save midi file, then convert into b64 format
    song_stream.write('midi', osp.join('resources', 'mid', filename + '.mid'))
    file = open(osp.join('resources', 'mid', filename+'.mid'), 'rb').read()
    b64 = base64.b64encode(file)

    return outStr.replace('\n', ''), b64


if __name__ == "__main__":
    classifier = Classifier()
    predict("./resources/sheets/test_piece2-1.png", classifier)
