import pickle

import numpy as np
from HMM import HiddenMarkovModel as hmm

# with open(r"C:\Users\rdela\Code Directory\CS4100\cs4100-final-proj\chord_data.pkl", "rb") as infile:
#     chords = pickle.load(infile)
#     chords.pop(26)
#     chords.pop(28)

with open("note_length_data.pkl", "rb") as infile:
    notes = pickle.load(infile)
    notes.pop(26)
    notes.pop(28)
    
for note in notes:
    print(note)
    print("\n")
# for piece in range(len(chords)):
#     if chords[piece][0] is False:
#         chords[piece].pop(0)
#         notes[piece].pop(0)
#     for measure in range(len(chords[piece])):
#         for chord in
        
        
    """4/4: 1 chord: 1 measure
    2 chords: 1 chord for 2 beats each
    3 chords: 1 chord for 2 beats, 2 chords for 1 beat each
    4 chords is 1 chord for each beat
    
    3/4: 1 chord: 1 measure
    2 chords: 1 chord for 2 beats, 1 chord for 1 beat
    3 chords: 3 chords for 1 beat each
    """
    
    # layer 1: notes     v
    # layer 2: rhythm      !=
    # layer 3: chords    ^


# array_list = []
# for i in range(len(chords)):
#     print(chords[i][1:])
#     chord_array = np.array(chords[i][1:])
#     print(chord_array.shape)
#     notes_array = np.array(notes[i])
#     print(notes_array.shape)

    # piece_array = np.vstack((), np.array(notes[i])))
    # array_list.append(piece_array)


# hm = hmm(
#     np.array(["H", "C"]),
#     observable_states,
#     transitions,
#     emissions,
#     start_probs,
#     end_probs,
# )
