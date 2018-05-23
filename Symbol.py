from enum import Enum


class NoteValue(Enum):
    UNKNOWN = 0
    REST = 1
    # whatever the music21/midi format needs? ( havent looked this up)
    A3 = 440


class SymbolType(Enum):
    UNKNOWN = 0
    TREBLE = 1
    BASS = 2
    NATURAL = 3
    FLAT = 4
    SHARP = 5
    SEMIQUAVER = 6
    QUAVER = 7
    QUATER = 8
    HALF = 9
    WHOLE = 10
    BARLINE = 11
    TIMESIG = 12
    MODDOT = 13  # as in either sticato or the extension
    SLUR = 14
    MODVOL = 15  # eg mp, p under staff
    # should rest be seperate or are they just whole/quater notes with value silence?


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

    def __init__(self, type, x, y, w, h):
        self.type = type  # type SymbolType
        self.symbol_order = 0  # not sure how to set this
        self.bar_num = 0        # For modifers that last the whole bar
        self.line_num = 0
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.im = None
        self.value = NoteValue.REST
        self.modifers = []  # if this is a note with modifers

    def work_out_type(self):
        pass

    def get_type(self):
        return self.type

    def is_note(self):
        return (self.type == SymbolType.SEMIQUAVER) or\
               (self.type == SymbolType.QUAVER) or\
               (self.type == SymbolType.QUATER) or\
               (self.type == SymbolType.HALF) or\
               (self.type == SymbolType.WHOLE)

    def is_bar(self):
        return self.type == SymbolType.BARLINE

    def is_modifer(self):
        return (self.type == SymbolType.TREBLE) or\
               (self.type == SymbolType.BASS) or\
               (self.type == SymbolType.SHARP) or\
               (self.type == SymbolType.FLAT) or\
               (self.type == SymbolType.NATURAL) or\
               (self.type == SymbolType.TIMESIG) or\
               (self.type == SymbolType.MODDOT) or \
               (self.type == SymbolType.MODVOL) or \
               (self.type == SymbolType.SLUR)

    def is_part_of_key_sig(self):
        return (self.type == SymbolType.TREBLE) or\
               (self.type == SymbolType.BASS) or\
               (self.type == SymbolType.SHARP) or\
               (self.type == SymbolType.FLAT) or\
               (self.type == SymbolType.NATURAL) or\
               (self.type == SymbolType.TIMESIG)  # im not sure about this one being a good idea
