import os
import pickle

import numpy as np

from HMM import HiddenMarkovModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "clean_data")
ET_FILE_PATH = os.path.join(BASE_DIR, "training")

with open(FILE_PATH + "/pair_to_map.pkl", "rb") as infile:
    pair_to_map = pickle.load(infile)
with open(FILE_PATH + "/chords_to_map.pkl", "rb") as infile:
    chords_to_map = pickle.load(infile)
with open(FILE_PATH + "/piece_mapped_chords.pkl", "rb") as infile:
    chords_by_piece = pickle.load(infile)
with open(FILE_PATH + "/piece_mapped_notes.pkl", "rb") as infile:
    notes_by_piece = pickle.load(infile)
with open(FILE_PATH + "/map_to_chords.pkl", "rb") as infile:
    map_to_chords = pickle.load(infile)
with open(FILE_PATH + "/map_to_pair.pkl", "rb") as infile:
    map_to_pair = pickle.load(infile)
with open(ET_FILE_PATH + "/transitions.csv", "r") as infile:
    transitions = np.genfromtxt(infile, delimiter=",")
with open(ET_FILE_PATH + "/emissions.csv", "r") as infile:
    emmissions = np.genfromtxt(infile, delimiter=",")

# test notes and durations
notes = [60, 64, 62, 67, 69, 70, 71, 72]
durations = [0.5, 0.5, 1.0, 0.25, 0.25, 0.5, 1.0, 0.5]


def standardize_octave(n):
    """standardize the octave of a note to be within the range of 60 to 71

    Args:
        n (int): note

    Returns:
        int : standardized note
    """
    return (n % 12) + 60


# gets note pairs assuming one chord for two notes, and duration of the pairs
def get_pairs(notes, durations):
    """get note pairs assuming one chord for two notes, and duration of the pairs

    Args:
        notes (list) : list of notes
        durations (list): list of durations

    Returns:
        pairs (list) : list of note pairs as tuples
        chord_durations (list) : list of durations of chords that should be played for each pair
    """
    if len(notes) % 2 == 0:
        pairs = [(notes[i], notes[i + 1]) for i in range(0, len(notes), 2)]
        chord_durations = [
            durations[i] + durations[i + 1] for i in range(0, len(durations), 2)
        ]
    else:
        pairs = [(notes[i], notes[i + 1]) for i in range(0, len(notes) - 3, 2)]
        pairs.append((notes[-3], notes[-1]))
        chord_durations = [
            durations[i] + durations[i + 1] for i in range(0, len(durations) - 3, 2)
        ]
        chord_durations.append(durations[-3] + durations[-2] + durations[-1])

    return pairs, chord_durations


def get_chord_sequence(notes, durations):
    """get chord sequence from notes and durations

    Args:
        notes (list) : list of notes
        durations (list) : list of durations

    Returns:
        dict : dictionary containing chords and chord durations
    """
    standardized_notes = list(standardize_octave(np.array(notes)))
    pairs, chord_durations = get_pairs(standardized_notes, durations)
    int_pairs = [pair_to_map[pair] for pair in pairs]

    all_chords = []
    for piece in chords_by_piece:
        all_chords += piece
    unique_chords = set(all_chords)
    n = len(unique_chords)
    all_notes = []
    for piece in notes_by_piece:
        all_notes += piece
    unique_notes = set(all_notes)

    firsts = [elem[0] for elem in chords_by_piece]
    lasts = [elem[-1] for elem in chords_by_piece]
    first_probs = np.array([firsts.count(i) / n for i in unique_chords])
    last_probs = np.array([lasts.count(i) / n for i in unique_chords])

    hmm = HiddenMarkovModel(
        np.array(list(unique_chords)),
        np.array(list(unique_notes)),
        transitions,
        emmissions,
        first_probs,
        last_probs,
    )

    # get chord sequence using HMM
    seq = hmm.predict_chords(int_pairs)

    # convert chord sequence to chord names
    chord_seq = [map_to_chords[s] for s in seq]

    midi_chord_info = {"chords": chord_seq, "chord_durations": chord_durations}


    print(f"Chords generated: {midi_chord_info["chords"]}")

    return midi_chord_info
