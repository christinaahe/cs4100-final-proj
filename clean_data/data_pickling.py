import pickle
import json
import numpy as np
from HMM import HiddenMarkovModel as hmm
from collections import Counter
import random

# Unpickle chord and note data made from mxlconverter.py
with open("chord_data.pkl", "rb") as infile:
    chords = pickle.load(infile)
    # Remove pieces with too much missing information
    chords.pop(43)
    chords.pop(30)

with open("note_rest_data.pkl", "rb") as infile:
    notes = pickle.load(infile)
    # Remove pieces with too much missing information
    notes.pop(43)
    notes.pop(30)


def get_beats_per_chord(num_chords, beats_per_measure):
    """
    Determines the distribution of beats per chord within a measure based on the number of chords and beats per measure.

    Parameters:
    - num_chords (int): The number of chords in the measure.
    - beats_per_measure (int): The total number of beats in the measure.

    Returns:
    - beats_per_chord (list): A list representing the distribution of beats per chord within the measure.
    """
    # If there is only one chord, assign all beats to it
    if num_chords == 1:
        return [beats_per_measure]
    elif beats_per_measure == 4:
        # Distribute beats evenly for four chords
        if num_chords == 4:
            return [1, 1, 1, 1]
        # Distribute beats as 2, 1, 1 for three chords
        elif num_chords == 3:
            return [2, 1, 1]
        # Distribute beats evenly for two chords
        elif num_chords == 2:
            return [2, 2]
    elif beats_per_measure == 3:
        # Distribute beats evenly for three chords
        if num_chords == 3:
            return [1, 1, 1]
        # Distribute beats as 2, 1 for two chords
        elif num_chords == 2:
            return [2, 1]
    elif beats_per_measure == 8:
        # Distribute beats evenly for eight chords
        if num_chords == 8:
            return [1] * 8
        # Distribute beats evenly for four chords
        elif num_chords == 4:
            return [2] * 4
        # Distribute beats evenly for two chords
        elif num_chords == 2:
            return [4, 4]
    elif beats_per_measure == 2:
        # Distribute beats evenly for two chords
        if num_chords == 2:
            return [1, 1]
    elif beats_per_measure == 1:
        # Distribute beats evenly for two chords
        if num_chords == 2:
            return [0.5, 0.5]
    elif beats_per_measure == 6:
        # Distribute beats evenly for three chords
        if num_chords == 3:
            return [2, 2, 2]
    else:
        # If no specific pattern matches, print a warning message
        print('Pattern not specified', beats_per_measure, num_chords)
        return None


def split_notes_by_chord(notes, beats_per_chord):
    """
    Splits a list of notes into groups representing individual chords based on specified beats per chord.

    Parameters:
    - notes (list): A list of tuples where each tuple contains a MIDI pitch (int) and its duration in beats (float).
    - beats_per_chord (list): A list of floats representing the desired number of beats for each chord.

    Returns:
    - notes_per_chord (list): A list of lists where each inner list contains the MIDI pitches of notes belonging to a chord.
    """
    # Initializes list for notes for each chord
    notes_per_chord = []
    # Initializes list for notes in current chord (inner list)
    notes_for_chord = []
    # Initializes a beat total and chord counter
    beat_tot = 0
    count = 0

    for n in notes:
        # Adds note to current chord notes list
        notes_for_chord.append(n[0])
        # Adds duration of current note to total beats counter
        beat_tot += n[1]
        # Check if beat count matches desired count for the chord
        if round(beat_tot, 3) == beats_per_chord[count]: # Round to avoid floating point errors
            # Appends current chord notes to overall list
            notes_per_chord.append(notes_for_chord)
            # Resets beat counter and current chord notes list
            beat_tot = 0
            notes_for_chord = []
            # Moves onto next chord
            count += 1
    return notes_per_chord


def clean_notes(notes):
    """
    Turns a list of MIDI note pitches corresponding to one chord into a list of only two notes with pitches transposed to the C3 octave.

    Parameters:
    - notes (list): A list of MIDI note pitches (integers), where rests are represented as -1.

    Returns:
    - cleaned_notes (list): A list of two MIDI note pitches, transposed to the C3 octave (MIDI note number 60).

    Note:
    - This function assumes that MIDI note pitches are represented as integers and rests are represented as -1.
    - If there is only one note, it duplicates it to ensure there are always at least two notes in the output.
    - If there are two notes, they are returned as is.
    - If there are three or more notes, only the first and third notes are returned.
    - If there are no notes remaining after cleaning (e.g., if all were rests), an empty list is returned.
    """
    # Removes rests and tranposes pitches to C3 octave
    notes_norests = [(n%12)+60 for n in notes if n != -1]
    # Calculate number of notes corresponding to a given chord
    num_notes = len(notes_norests)

    # Determines pair of notes differently based on number of notes for given chord
    if num_notes == 1:
         # If there is only one note, duplicate it
        return notes_norests * 2
    if num_notes == 2:
         # If there are two notes, return as is
        return notes_norests
    if num_notes >= 3:
         # If there are three or more notes, return the first and third note
        return [notes_norests[0], notes_norests[2]]
    # If there are no notes remaining (after removing rests), return an empty list
    return []


def tuple_notes(notes):
    """
    Converts a list of lists of lists containing MIDI note pitches into a list of tuples.

    Parameters:
    - notes (list): A list of lists of lists where each inner list corresponds to a piece and contains a list of tuples representing MIDI note pitches.

    Returns:
    - final (list): A list that is the same as the input notes list except the innermost list is a tuple instead.
    """
    final = []
    # Iterate through each piece of music represented as a list of note pairs
    for piece in notes:
        # Convert each note pair into a tuple and append it to the final list
        final.append([tuple(pair) for pair in piece])
    return final


def clean_data(chords, notes):
    """
    Cleans chord and note data, removing invalid measures and processing chords and notes for each measure.

    Parameters:
    - chords (list): A list of lists where each inner list represents chords for a piece of music.
    - notes (list): A list of lists where each inner list represents notes for a piece of music.

    Returns:
    - all_chords (list): A list of lists where each inner list contains one cleaned chord for a piece of music.
    - all_notes (list): A list of tuples where each tuple represents a note pair corresponding to one chord for a piece of music.

    """
    # Initialize lists to store cleaned chord and note data for all pieces
    all_chords = []
    all_notes = []

    # Iterate through each piece
    for piece in range(len(chords)):
        # Initialize lists to sore cleaned chord and note data for current piece
        cleaned_chords = []
        cleaned_notes = []

        # Remove first measure if there is no chord data (removes measure for notes as well)
        # usually occurs if the first measure contains a pickup note(s)
        if chords[piece][0] is False:
            chords[piece].pop(0)
            notes[piece].pop(0)
        # Removes measures where there are no chord data (removes measure for notes as well)
        if [0] in chords[piece] or [] in chords[piece]:
            # Find indices of missing chord measure
            indexes = [i for i, x in enumerate(chords[piece]) if x == [0] or x == []]
            # Removes given indices
            for index in sorted(indexes, reverse=True):
                chords[piece].pop(index)
                notes[piece].pop(index)

        # Iterates through each measure of the piece
        for measure in range(len(chords[piece])):
            # Gets the number of beats in the measure
            beats_per_measure = round(sum([dur for _, dur in notes[piece][measure]]), 3)
            # Gets chords and notes in current measure
            curr_chords = chords[piece][measure]
            curr_notes = notes[piece][measure]
            # Calculates number of chords in the measure
            num_chords = len(curr_chords)

            # Gets the beat distribution for the chords in the measure
            beats_per_chord = get_beats_per_chord(num_chords, beats_per_measure)
            # Skip to next measure if beat distribution pattern is not defined in get_beats_per_chord (only contains most common beat patterns)
            if beats_per_chord is None:
                continue
            # Splits notes by chord (list of list of notes)
            notes_per_chord = split_notes_by_chord(curr_notes, beats_per_chord)
            
            # Initializes lists for cleaned notes per chord and the chord itself
            clean_notes_per_chord = []
            clean_chords = []
            for i, notes_pc in enumerate(notes_per_chord):
                # Clean notes for the given notes in the chord
                clean = clean_notes(notes_pc)
                if clean == []:
                    continue
                clean_notes_per_chord.append(clean)
                clean_chords.append([curr_chords[i]])

            # Add cleaned notes and chords for the current measure to the rest of the cleaned notes
            cleaned_notes = cleaned_notes + clean_notes_per_chord
            cleaned_chords = cleaned_chords + clean_chords
        # Append cleaned chord and note data for the current piece to the respective lists
        all_chords.append(cleaned_chords)
        all_notes.append(cleaned_notes)

    return all_chords, tuple_notes(all_notes)


def join_pieces(data):
    """
    Joins multiple pieces of data into a single list.

    Parameters:
    - data (list): A list of lists.

    Returns:
    - final (list): A single list containing all the inner lists joined together.
    """
    final = []
    for piece in data:
        final = final + piece
    return final


def get_mapping_dicts(data):
    """
    Generates mapping dictionaries between chords/notes and integers.

    Parameters:
    - data (list): A list containing chords/notes.

    Returns:
    - map_to_pair (dict): A dictionary mapping integers to unique chords/notes.
    - pair_to_map (dict): A dictionary mapping chords/notes to integers.
    """
    # Get unique elements and shuffle them
    unique_data = list(set(data))
    random.shuffle(unique_data)
    # Create a dictionary mapping integers to unique elements
    map_to_pair = dict(enumerate(unique_data))
    # Create a dictionary mapping unique elements to integers
    pair_to_map = {val: key for key, val in map_to_pair.items()}
    return map_to_pair, pair_to_map


def main():
    # Get cleaned chords and note pairs separated by piece
    piece_chords, piece_notes = clean_data(chords, notes)
    # Join chords and note pairs from all pieces into one
    separated_chords = join_pieces(piece_chords)
    separated_notes = join_pieces(piece_notes)

    # Generate dictionary mappings for note pairs to integers
    map_to_pair, pair_to_map = get_mapping_dicts(separated_notes)
    # Turn piece separated and joined note pairs into integer mappings
    piece_mapped_notes = [[pair_to_map[pair] for pair in piece] for piece in piece_notes]
    separated_mapped_notes = join_pieces(piece_mapped_notes)

    # Generate dictionary mappings for chords to integers
    map_to_chords, chords_to_map = get_mapping_dicts([lst[0] for lst in separated_chords])
    # Turn piece separated and joined chords into integer mappings
    piece_mapped_chords = [[chords_to_map[chord] for chord in piece] for piece in [[c[0] for c in p] for p in piece_chords]]
    separated_mapped_chords = join_pieces(piece_mapped_chords)

    # Pickle and store cleaned data and mappings
    # pickles = {'map_to_pair': map_to_pair, 'pair_to_map': pair_to_map, 'piece_mapped_notes': piece_mapped_notes,
    #            'separated_mapped_notes': separated_mapped_notes, 'map_to_chords': map_to_chords,
    #            'chords_to_map': chords_to_map,
    #            'piece_mapped_chords': piece_mapped_chords, 'separated_mapped_chords': separated_mapped_chords}
    # for name, dict in pickles.items():
    #     with open(name + '.pkl', 'wb') as f:
    #         pickle.dump(dict, f)

if __name__ == '__main__':
    main()
