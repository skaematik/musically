from music21 import *
import numpy as np


def gen_tinynotation(n_bars=10, options=None):
    """Generates tinyNotation music string. Assumes x/4 time signature.
    
    n_bars: how many bars of music to generate
    options: dict of preference weights
        options["note_weights"] is a list of weights for each possible note
        options["time_weights"] is a list of weights for each possible time for each note
    
    """
    # bottom number in time signature
    per_beat = 4
    
    # top number in time signature
    per_measure_range = [2,3,4]
    per_measure = np.random.choice(per_measure_range)
    
    # note range
    note_range = ["c", "d", "e", "f", "g", "a", "b", "c'", "d'", "e'", "f'", "g'", "a'", "b'", "r"]
    
    # time range for each note
    time_range = [1., 1./2, 1./4, 1./8]
    
    if options is None:
        note_weights = [1]*len(note_range)
        time_weights = [1]*len(time_range)
    else:
        note_weights = options["note_weights"]
        time_weights = options["time_weights"]
    
    note_weights = np.array(note_weights)
    time_weights = np.array(time_weights)
    note_weights = note_weights.astype(float) / np.sum(note_weights)
    time_weights = time_weights.astype(float) / np.sum(time_weights)
    
    def sample():
        note = np.random.choice(note_range, p=note_weights)
        time = np.random.choice(time_range, p=time_weights)
        return note, time * per_beat
    
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
            string += note + str(int(per_beat/time)) + " "
            if time_offset == per_measure:
                break
                    
        return string.strip()
    
    
    string = " ".join([generate_bar() for _ in range(n_bars)])
    
    return str(per_measure) + "/" + str(per_beat) + " " + string


if __name__ == "__main__":
    note_weights = [1] * 15
    note_weights[0] = 10
    
    time_weights = [1] * 4
    
    options = {}
    options["note_weights"] = note_weights
    options["time_weights"] = time_weights
    musc = gen_tinynotation(options=options)
    print(musc)
    stream = converter.parse("tinynotation: " + musc)
    stream.show("text")
    stream.show()
