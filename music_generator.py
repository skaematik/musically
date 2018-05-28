from music21 import *
import numpy as np
from music21.converter.subConverters import ConverterMusicXML


class MusicGenerator:
    # bottom number in time signature
    per_beat = 4

    # note range
    note_range = ["c", "d", "e", "f", "g", "a", "b",
                  "c'", "d'", "e'", "f'", "g'", "a'", "b'", "r"]

    # time range for each note
    time_range = [1, 2, 4, 8]

    def __init__(self, options=None):
        """
        options: dict of preference weights
            options["note_weights"] is a list of weights for each possible note
            options["time_weights"] is a list of weights for each possible time for each note
        """
        if options is None:
            note_weights = [1]*len(note_range)
            time_weights = [1]*len(time_range)
        else:
            note_weights = options["note_weights"]
            time_weights = options["time_weights"]

        note_weights = np.array(note_weights)
        time_weights = np.array(time_weights)
        self.note_weights = note_weights.astype(float) / np.sum(note_weights)
        self.time_weights = time_weights.astype(float) / np.sum(time_weights)

    def tinynotation_v2(self, n_bars=10, per_measure=None):
        """Generates tinyNotation music string. Assumes x/4 time signature.

        n_bars: how many bars of music to generate

        """
        # top number in time signature
        per_measure_range = [2, 3, 4]
        if per_measure is None:
            per_measure = np.random.choice(per_measure_range)
        full_length = {2: 0.5, 3: 0.75, 4: 1}

        def generate_bar():
            bar = ""
            bar_length = 0.
            total_bar_len = full_length[per_measure]
            init_note = np.random.choice(self.note_range, p=self.note_weights)
            note_idx = self.note_range.index(init_note)
            while True:  # pick length that fits in bar
                init_length = np.random.choice(self.time_range, p=self.time_weights)
                new_len = (bar_length + 1 / init_length)
                if new_len <= total_bar_len:
                    break
            bar += "{}{} ".format(init_note, init_length)
            bar_length += 1/init_length
            while bar_length < full_length[per_measure]:
                new_note_idx = np.random.normal(loc=note_idx,scale=3)
                if new_note_idx < 0:
                    new_note_idx *= -1
                while new_note_idx >= len(self.note_range):
                    new_note_idx -= (new_note_idx-len(self.note_range))*2 # reflect back
                if new_note_idx < 0:
                    new_note_idx = 0
                if new_note_idx >= len(self.note_range):
                    new_note_idx = len(self.note_range)-1
                note_idx = new_note_idx
                note = self.note_range[int(note_idx)]
                while True: # pick length that fits in bar
                    length = np.random.choice(self.time_range, p=self.time_weights)
                    new_len = (bar_length + 1 / length)
                    if new_len <= total_bar_len:
                        break
                bar += "{}{} ".format(note, length)
                bar_length += 1 / length

            return bar

        string = " ".join([generate_bar() for _ in range(n_bars)])

        return str(per_measure) + "/" + str(self.per_beat) + " " + string

    def tinynotation(self, n_bars=10):
        """Generates tinyNotation music string. Assumes x/4 time signature.

        n_bars: how many bars of music to generate

        """
        # top number in time signature
        per_measure_range = [2, 3, 4]
        per_measure = np.random.choice(per_measure_range)

        def sample():
            note = np.random.choice(self.note_range, p=self.note_weights)
            time = np.random.choice(self.time_range, p=self.time_weights)
            return note, time * self.per_beat

        def generate_bar():
            string = ""
            time_offset = 0.

            sample_count = 0
            sample_limit = 10

            while True:
                note, time = sample()

                if time_offset + time <= per_measure:
                    sample_count = 0
                else:
                    sample_count += 1
                    if sample_count == sample_limit:
                        time = per_measure - time_offset
                    else:
                        continue

                time_offset += time
                string += note + str(int(self.per_beat/time)) + " "
                if time_offset == per_measure:
                    break
            return string.strip()

        string = " ".join([generate_bar() for _ in range(n_bars)])

        return str(per_measure) + "/" + str(self.per_beat) + " " + string


def gen_pieces(n):
    for i in range(n):
        note_weights = [1] * len(MusicGenerator.note_range)
        note_weights[0] = 1

        time_weights = [1] * len(MusicGenerator.time_range)

        options = {}
        options["note_weights"] = note_weights
        options["time_weights"] = time_weights

        generator = MusicGenerator(options=options)
        musc = generator.tinynotation(n_bars=2000)

        print(musc)
        stream = converter.parse("tinynotation: " + musc)
        # stream.show("text")
        # stream.show("musicxml")
        conv_musicxml = ConverterMusicXML()
        scorename = 'myScoreName{:03}.xml'.format(i)
        filepath = './tests/' + scorename
        out_filepath = conv_musicxml.write(
            stream, 'musicxml', fp=filepath, subformats=['png'])


if __name__ == "__main__":
    # gen_pieces(1)
    note_weights = [1] * len(MusicGenerator.note_range)
    note_weights[0] = 1

    time_weights = [1] * len(MusicGenerator.time_range)

    options = {}
    options["note_weights"] = note_weights
    options["time_weights"] = time_weights

    generator = MusicGenerator(options=options)
    musc = generator.tinynotation_v2(n_bars=20, per_measure=4)

    print(musc)
    stream = converter.parse("tinynotation: " + musc)
    stream.show("text")
    stream.show("musicxml")
