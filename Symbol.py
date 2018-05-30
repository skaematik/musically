from operator import itemgetter

import cv2
import numpy as np

from enum import Enum


class NoteValue(Enum):
    UNKNOWN = 0
    REST = 1
    # whatever the music21/midi format needs? ( havent looked this up)
    A3 = 440


SymbolType_Names = ['barline', 'clef', 'eighth', 'half', 'quarter', 'rest_eighth', 'rest_measure', 'rest_quarter',
                    'tied_eighth', 'timeSig', 'whole']

SymbolType_Duration = {'eighth': 0.5, 'half': 2, 'quarter': 1, 'rest_eighth': 0.5, 'rest_measure': 4, 'rest_quarter': 1,
                       'tied_eighth': 1, 'whole': 4}


class SymbolType(Enum):
    UNKNOWN = -1
    BARLINE = 0
    CLEF = 1
    EIGHTH = 2
    HALF = 3
    QUARTER = 4
    REST_EIGHTH = 5
    REST_MEASURE = 6  # could also be half
    REST_QUARTER = 7
    TIED_EIGHTH = 8
    TIMESIG = 9
    WHOLE = 10
    # future?
    BASS = 11
    NATURAL = 12
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

    def __init__(self, type, x, y, w, h, line_number, staff_lines=[]):
        self.markHalf = False
        self.markFull = False
        self.type = type  # type SymbolType
        self.symbol_order = 0  # not sure how to set this
        self.bar_num = 0  # For modifers that last the whole bar
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

    def work_out_type(self, label_id):
        self.type = SymbolType(label_id)
        print(self.type.name)

    def get_type(self):
        return self.type

    def get_name(self):
        if 0 <= self.type.value <= 10:
            return SymbolType_Names[self.type.value]
        return 'UNKNOWN'

    def determine_pitch(self, templates, also_return_offsets=False, image=None):
        """ determines pitch of note, and x_offset at which it occurs"""
        if image is None:
            image = self.im

        results = []
        for tem_tup in templates:
            match = cv2.matchTemplate(image, tem_tup[0], method=cv2.TM_CCOEFF)
            (minV, maxV, minLoc, maxLoc) = cv2.minMaxLoc(match)
            results.append((maxLoc, maxV, tem_tup[1], tem_tup[0].shape))
        results = sorted(results, key=itemgetter(1))
        if self.type == SymbolType.HALF and results[-1][2].startswith('full'):
            self.markFull = True
            print('full mark')
        if (self.type == SymbolType.QUARTER or self.type == SymbolType.EIGHTH) and results[-1][2].startswith('half'):
            self.markHalf = True
            print('half mark')
        # print(results[-1])
        middle = self.y + ((results[-1][0][1] * 2) - self.offsety) + results[-1][3][0]
        x_offset = self.x + ((results[-1][0][0] * 2) - self.offsetx) + results[-1][3][1]

        x_idx_4 = -1  # Find where in line the note is
        x_idx_2 = -1  # Find where in line the note is
        x_idx_0 = -1  # Find where in line the note is
        for i in range(max(len(self.staff_lines[0]),len(self.staff_lines[2]),len(self.staff_lines[4]))):
            if i < len(self.staff_lines[0]) and self.staff_lines[0][i][0] > self.x and x_idx_0 == -1:
                x_idx_0 = i
            if i < len(self.staff_lines[2]) and self.staff_lines[2][i][0] > self.x and x_idx_2 == -1:
                x_idx_2 = i
            if i < len(self.staff_lines[4]) and self.staff_lines[4][i][0] > self.x and x_idx_4 == -1:
                x_idx_4 = i
            if x_idx_0 != -1 and x_idx_2 != -1 and x_idx_4 != -1:
                break
        if x_idx_0 == -1:
            x_idx_0 = len(self.staff_lines[0])-1
        if x_idx_2 == -1:
            x_idx_2 = len(self.staff_lines[2]) - 1
        if x_idx_4 == -1:
            x_idx_4 = len(self.staff_lines[4]) - 1

        if middle > self.staff_lines[4][x_idx_4][1]:  # Below the final line
            amount_below = middle - self.staff_lines[4][x_idx_4][1]
            amount_below = amount_below / self.staff_white
            amount = round(amount_below * 2.0) / 2.0
            amount = - (amount * 2)
        elif middle < self.staff_lines[0][x_idx_0][1]:  # above first line
            amount_above = self.staff_lines[0][x_idx_0][1] - middle
            amount_above = amount_above / self.staff_white
            amount = round(amount_above * 2.0) / 2.0
            amount = amount * 2 + 8
        else:
            amount_above_below = self.staff_lines[2][x_idx_2][1] - middle
            amount_above_below = amount_above_below / self.staff_white
            amount = round(amount_above_below * 2.0) / 2.0
            amount = amount * 2 + 4
        if also_return_offsets:
            return amount, results[-1][0], results[-1][3], results[-1][1]
        return amount

    def is_note(self):
        return (self.type == SymbolType.EIGHTH) or \
               (self.type == SymbolType.HALF) or \
               (self.type == SymbolType.QUARTER) or \
               (self.type == SymbolType.TIED_EIGHTH) or \
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
        return (self.type == SymbolType.CLEF) or \
               (self.type == SymbolType.BASS) or \
               (self.type == SymbolType.SHARP) or \
               (self.type == SymbolType.FLAT) or \
               (self.type == SymbolType.NATURAL) or \
               (self.type == SymbolType.TIMESIG)  # im not sure about this one being a good idea

    def determine_beamed_pitch(self, templates):
        if self.type != SymbolType.TIED_EIGHTH:
            return []
        image = np.array(self.im)

        note, location, size, best_max = self.determine_pitch(templates, also_return_offsets=True, image=image)
        value = best_max
        image[location[1]:(location[1] + size[1]), location[0]:(location[0] + size[0])] = 255
        result = [(note, location[0])]
        while value / best_max > 0.9:
            note, location, size, value = self.determine_pitch(templates, also_return_offsets=True, image=image)
            if value / best_max < 0.9:
                break
            image[location[1]:(location[1] + size[1]), location[0]:(location[0] + size[0])] = 255
            result.append((note, location[0]))
        return [n[0] for n in sorted(result, key=itemgetter(1))]

    def get_rest_duration(self):
        if not self.is_rest():
            return 0
        else:
            if self.type == SymbolType.REST_EIGHTH:
                return 0.5
            if self.type == SymbolType.REST_QUARTER:
                return 1
            x_idx = len(self.staff_lines[2]) - 1
            for i in range(len(self.staff_lines[2])):
                if self.staff_lines[2][i][0] > self.x:
                    x_idx = i
                    break
            if self.y + self.h - self.staff_lines[2][x_idx][1] > 4 * self.staff_black:
                return 4  # full measure rest
            return 2  # half measure rest

