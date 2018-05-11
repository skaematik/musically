from music21 import *
import numpy as np


def gen_tinynotation(n_bars=10):
    """Generates tinyNotation music string. Assumes x/4 time signature.
    
    n_bars: how many bars of music to generate
    
    """
    # bottom number in time signature
    per_beat = 4
    
    # top number in time signature
    per_measure_opts = [2,3,4]
    per_measure = per_measure_opts[np.random.randint(len(per_measure_opts))]
    
    def sample():
        notes = ["c", "d", "e", "f", "g", "a", "b", "c'", "d'", "e'", "f'", "g'", "a'", "b'", "r"]
        times = [1., 1./2, 1./4, 1./8]
        
        note = notes[np.random.randint(len(notes))]
        time = times[np.random.randint(len(times))]
        
        return note, time * per_beat
    
    def generate_bar():
        string = ""
        time_offset = 0.
        
        while True:
            note, time = sample()
            if time_offset + time <= per_measure:
                time_offset += time
                string += note + str(int(per_beat/time)) + " "
                if time_offset == per_measure:
                    break
        return string.strip()
        
    
    string = " ".join([generate_bar() for _ in range(n_bars)])
    
    return str(per_measure) + "/" + str(per_beat) + " " + string


if __name__ == "__main__":
    msc = gen_tinynotation()
    print(msc)
    stream = converter.parse("tinynotation: " + msc)
    stream.show("text")
    stream.show()
