import cv2
import os
from typing import List, Any

from music21 import *

from Symbol import Symbol, SymbolType, SymbolType_Duration


def get_duration_from_name(name):
    d = duration.Duration()
    d.quarterLength = SymbolType_Duration[name]
    return d


treble_values = {
    -8: 'D3',
    -7: 'E3',
    -6: 'F3',
    -5: 'G3',
    -4: 'A3',
    -3: 'B3',
    -2: 'C4',
    -1: 'D4',
    0: 'E4',  # Bottom line
    1: 'F4',
    2: 'G4',
    3: 'A4',
    4: 'B4',
    5: 'C5',
    6: 'D5',
    7: 'E5',
    8: 'F5',  # Top line
    9: 'G5',
    10: 'A5',
    11: 'B5',
    12: 'C6',
    13: 'D6',
    14: 'E6'
}

def to_pitchStr_treble(amount):
    try :
        return treble_values[amount]
    except Exception:
        return 'A2'


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
        bar = 1;
        bar_length = 0
        i = 0
        for sym in self.symbols:
            if sym.is_bar():
                print('bar {} is {} long'.format(bar, bar_length))
                bar += 1
                if bar_length < 4:
                    need = 4 - bar_length
                    d = duration.Duration()
                    d.quarterLength = need
                    print('bar {} is too long adding {}'.format(bar, need))
                    rest = note.Rest(duration=d)
                    self.stream.append(rest)
                if bar_length > 4:
                    need = 4 - (bar_length - 4)
                    d = duration.Duration()
                    d.quarterLength = need
                    print('bar {} is too long short {}'.format(bar, need))
                    rest = note.Rest(duration=d)
                    self.stream.append(rest)
                bar_length = 0
            if sym.is_part_of_key_sig():
                if sym.get_type() == SymbolType.CLEF:
                    pass
                if sym.get_type() == SymbolType.TIMESIG:
                    pass
            if sym.is_note():
                if sym.get_type() == SymbolType.TIED_EIGHTH:
                    beam_notes = sym.determine_beamed_pitch([t for t in filter(lambda t: t[1] == 'full.png', templates)])
                    for n_p in beam_notes:
                        pitch_str = to_pitchStr_treble(n_p)
                        n = note.Note(pitchName=pitch_str, duration=get_duration_from_name('eighth'))
                        bar_length += 0.5
                        self.stream.append(n)
                        # Music21 will decided a nice beaming scheme for us
                else:
                    pitch_str = to_pitchStr_treble(sym.determine_pitch(templates))
                    d = get_duration_from_name(sym.get_name())
                    bar_length += d.quarterLength
                    n = note.Note(pitchName=pitch_str, duration=d)
                    self.stream.append(n)
            if sym.is_rest():
                d = sym.get_rest_duration()
                bar_length += d
                dur = duration.Duration()
                dur.quarterLength = d
                n = note.Rest(duration=dur)
                self.stream.append(n)


