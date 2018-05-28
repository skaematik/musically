import cv2
import os
from typing import List, Any

from music21 import *

from Symbol import Symbol, SymbolType

duration_len = {
    'eighth': 0.5,
    'half': 2.0,
    'quarter': 1.0,
    'rest_eighth': 0.5,
    'rest_measure': 2.0,
    'rest_quarter': 1.0,
    'tied_eighth': 1,
    'whole': 4}


def get_duration_from_name(name):
    d = duration.Duration()
    d.quarterLength = duration_len[name]
    return d


class Song:
    """ Collection of symbols making up a song, could be over multiple pages"""
    symbols: List[Symbol]

    def __init__(self):
        self.symbols = []
        self.stream = stream.Stream()

    def add_symbols(self, symbols):
        """ add sets of symbols if multipage song with seperate segmenter classes"""
        self.symbols.extend(symbols)

    def parse_symbols(self):
        """Go through notes in order and check for errors, later this will apply modifiers"""
        template_names = sorted(os.listdir('./resources/templates/'))
        templates = []
        for name in template_names:
            if name != '.DS_Store':
                tem = cv2.imread('./resources/templates/'+name, 0)
                templates.append((tem, name))

        for sym in self.symbols:
            if sym.is_part_of_key_sig():
                if sym.get_type() == SymbolType.CLEF:
                    pass
                if sym.get_type() == SymbolType.TIMESIG:
                    pass
            if sym.is_note():
                sym.determine_pitch(templates)
                if sym.get_type() == SymbolType.TIED_EIGHTH:
                    n1 = note.Note(pitchName='C4', duration=get_duration_from_name('eighth'))
                    n2 = note.Note(pitchName='C4', duration=get_duration_from_name('eighth'))
                    n1.beams.fill('eighth', type='start')
                    n2.beams.fill('eighth', type='stop')
                    self.stream.append(n1)
                    self.stream.append(n2)
                else:
                    n = note.Note(pitchName='C4', duration=get_duration_from_name(sym.get_name()))
                    self.stream.append(n)
            if sym.is_rest():
                n = note.Rest(duration=get_duration_from_name(sym.get_name()))
                self.stream.append(n)
        self.stream.show()
