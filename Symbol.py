from operator import itemgetter

import cv2
import numpy as np

from enum import Enum



class NoteValue(Enum):
    UNKNOWN = 0
    REST = 1
    # whatever the music21/midi format needs? ( havent looked this up)
    A3 = 440

class SymbolType(Enum):
    names_ = ['barline', 'clef', 'eighth', 'half', 'quarter', 'rest_eighth', 'rest_measure', 'rest_quarter', 'tied_eighth', 'timeSig', 'whole']
    UNKNOWN = -1
    BARLINE = 0
    CLEF = 1
    EIGHTH = 2
    HALF = 3
    QUARTER = 4
    REST_EIGHTH = 5
    REST_MEASURE = 6 #could also be half
    REST_QUARTER = 7
    TIED_EIGHTH = 8
    TIMESIG = 9
    WHOLE = 10
    #future?
    BASS = 11
    NATURAL =12
    FLAT = 14
    SHARP = 15
    SEMIQUAVER = 16
    MODDOT = 13  # as in either sticato or the extension
    SLUR = 14
    MODVOL = 15  # eg mp, p under staff


class Symbol:
    """Class for each symbol in text
        type  SymbolType:   type of the symbol
        id int:             ?? id?
        symbol_order int:   order of note/symbol in overall piece
        bar_num int:        bar that symbol is a part of or -1 if not part of bar
        line_num int:       line number that symbol is part of or -1 if not part of line
        x int:              x of top right
        y int:              y of top right
        w int:              width
        h int:              height
        value NoteValue:    value of symbol or NoteValue.UNKNOWN if not applicable
        modifers list:      list of the ids of any symbols that modifer this symbols value
        """

    def __init__(self, type, x, y, w, h,line_number, staff_lines=[]):
        self.type = type  # type SymbolType
        self.symbol_order = 0  # not sure how to set this
        self.bar_num = 0        # For modifers that last the whole bar
        self.line_num = line_number
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.im = None
        self.offsetx = 0
        self.offsety = 0
        self.staff_white = 0
        self.staff_black = 0
        self.staff_lines = staff_lines
        self.value = NoteValue.REST
        self.modifers = []  # if this is a note with modifers

    def work_out_type(self,label_id):
        self.type = SymbolType(label_id)

    def get_type(self):
        return self.type

    def get_name(self):
        if 0 <= self.type.value <= 10:
            return SymbolType.names_.value[self.type.value]
        return 'UNKNOWN'

    def determine_pitch(self, templates):
        results = []
        for tem_tup in templates:
            match = cv2.matchTemplate(self.im,tem_tup[0], method=cv2.TM_CCOEFF)
            (minV, maxV, minLoc, maxLoc) = cv2.minMaxLoc(match)
            results.append((maxLoc, maxV, tem_tup[1], tem_tup[0].shape))
        results = sorted(results, key=itemgetter(1))
        # print(results[-1])
        middle = self.y+((results[-1][0][1]*2)-self.offsety) + results[-1][3][0] #last one maybe wrong?
        x_idx = -1
        for i in range(len(self.staff_lines[0])):
            if self.staff_lines[0][i][0] > self.x:
                x_idx = i
                break
        if middle > self.staff_lines[4][x_idx][1]:  # Below the final line
            amount_below = middle - self.staff_lines[4][x_idx][1]
            amount_below = amount_below / self.staff_white
            amount = round(amount_below * 2.0) / 2.0
            print('below: {}'.format(amount))
            amount = - (amount*2)
        elif middle < self.staff_lines[0][x_idx][1]:  # above first line
            amount_above = self.staff_lines[0][x_idx][1] - middle
            amount_above = amount_above / self.staff_white
            amount = round(amount_above * 2.0) / 2.0
            print('above: {}'.format(amount))
            amount = amount*2 + 8
        else:
            amount_above_below = self.staff_lines[2][x_idx][1] - middle
            amount_above_below = amount_above_below / self.staff_white
            amount = round(amount_above_below * 2.0) / 2.0
            print('between: {}'.format(amount))
            amount = amount * 2 + 4
        print('amount: {}'.format(amount))
        return amount
        # check above
            # how above
        # where check inside
        return


    def is_note(self):
        return (self.type == SymbolType.EIGHTH) or\
               (self.type == SymbolType.HALF) or\
               (self.type == SymbolType.QUARTER) or\
               (self.type == SymbolType.TIED_EIGHTH) or\
               (self.type == SymbolType.WHOLE)

    def is_rest(self):
        return (self.type == SymbolType.REST_EIGHTH) or \
               (self.type == SymbolType.REST_MEASURE) or \
               (self.type == SymbolType.REST_QUARTER)

    def is_bar(self):
        return self.type == SymbolType.BARLINE

    def is_modifer(self):
        return False

    def is_part_of_key_sig(self):
        return (self.type == SymbolType.CLEF) or\
               (self.type == SymbolType.BASS) or\
               (self.type == SymbolType.SHARP) or\
               (self.type == SymbolType.FLAT) or\
               (self.type == SymbolType.NATURAL) or\
               (self.type == SymbolType.TIMESIG)  # im not sure about this one being a good idea
