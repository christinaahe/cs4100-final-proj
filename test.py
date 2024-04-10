import pickle
import json
import numpy as np
from HMM import HiddenMarkovModel as hmm

with open("chord_data.pkl", "rb") as infile:
    chords = pickle.load(infile)
    chords.pop(43)
    chords.pop(30)

with open("note_rest_data.pkl", "rb") as infile:
    notes = pickle.load(infile)
    notes.pop(43)
    notes.pop(30)


def get_beats_per_chord(num_chords, beats_per_measure):
    if num_chords == 1:
        return [beats_per_measure]
    elif beats_per_measure == 4:
        if num_chords == 4:
            return [1, 1, 1, 1]
        elif num_chords == 3:
            return [2, 1, 1]
        elif num_chords == 2:
            return [2, 2]
    elif beats_per_measure == 3:
        if num_chords == 3:
            return [1, 1, 1]
        elif num_chords == 2:
            return [2, 1]
    elif beats_per_measure == 8:
        if num_chords == 8:
            return [1] * 8
        elif num_chords == 4:
            return [2] * 4
        elif num_chords == 2:
            return [4, 4]
    elif beats_per_measure == 2:
        if num_chords == 2:
            return [1, 1]
    elif beats_per_measure == 1:
        if num_chords == 2:
            return [0.5, 0.5]
    elif beats_per_measure == 6:
        if num_chords == 3:
            return [2, 2, 2]
    else:
        print('Pattern not specified', beats_per_measure, num_chords)
        return None


def split_notes_by_chord(notes, beats_per_chord):
    notes_per_chord = []

    count = 0
    beat_tot = 0
    notes_for_chord = []

    for n in notes:
        beat_tot += n[1]
        notes_for_chord.append(n[0])
        if round(beat_tot, 3) == beats_per_chord[count]:
            notes_per_chord.append(notes_for_chord)
            beat_tot = 0
            notes_for_chord = []
            count += 1
    return notes_per_chord


def clean_notes(notes):
    notes_norests = [n for n in notes if n != -1]
    num_notes = len(notes_norests)
    if num_notes == 1:
        return notes_norests * 2
    if num_notes == 2:
        return notes_norests
    if num_notes >= 3:
        return [notes_norests[0], notes_norests[2]]
    return []


def tuple_notes(notes):
    final = []
    for piece in notes:
        final.append([tuple(pair) for pair in piece])
    return final


def clean_data(chords, notes):
    all_chords = []
    all_notes = []
    for piece in range(len(chords)):
        cleaned_chords = []
        cleaned_notes = []

        # remove measures with no chord data
        if chords[piece][0] is False:
            chords[piece].pop(0)
            notes[piece].pop(0)

        if [0] in chords[piece] or [] in chords[piece]:
            indexes = [i for i, x in enumerate(chords[piece]) if x == [0] or x == []]
            for index in sorted(indexes, reverse=True):
                chords[piece].pop(index)
                notes[piece].pop(index)

        for measure in range(len(chords[piece])):
            beats_per_measure = round(sum([dur for _, dur in notes[piece][measure]]), 3)
            curr_chords = chords[piece][measure]
            curr_notes = notes[piece][measure]
            num_chords = len(curr_chords)

            beats_per_chord = get_beats_per_chord(num_chords, beats_per_measure)
            if beats_per_chord is None:
                continue
            notes_per_chord = split_notes_by_chord(curr_notes, beats_per_chord)

            clean_notes_per_chord = []
            clean_chords = []
            for i, notes_pc in enumerate(notes_per_chord):
                clean = clean_notes(notes_pc)
                if clean == []:
                    continue
                clean_notes_per_chord.append(clean)
                clean_chords.append([curr_chords[i]])

            cleaned_notes = cleaned_notes + clean_notes_per_chord
            cleaned_chords = cleaned_chords + clean_chords
        all_chords.append(cleaned_chords)
        all_notes.append(cleaned_notes)

    return all_chords, tuple_notes(all_notes)


def join_pieces(data):
    final = []
    for piece in data:
        final = final + piece
    return final


# def unique_notes(notes):
#     all = []
#     for piece in notes:
#         for pair in piece:
#             all = all + pair
#     return set(all)

def get_mapping_dicts(data):
    map_to_pair = dict(enumerate(set(data)))
    pair_to_map = {val: key for key, val in map_to_pair.items()}
    return map_to_pair, pair_to_map


piece_chords, piece_notes = clean_data(chords, notes)

separated_chords = join_pieces(piece_chords)
separated_notes = join_pieces(piece_notes)

map_to_pair, pair_to_map = get_mapping_dicts(separated_notes)
piece_mapped_notes = [[pair_to_map[pair] for pair in piece] for piece in piece_notes]
separated_mapped_notes = join_pieces(piece_mapped_notes)

map_to_chords, chords_to_map = get_mapping_dicts([lst[0] for lst in separated_chords])
piece_mapped_chords = [[chords_to_map[chord] for chord in piece] for piece in [[c[0] for c in p] for p in piece_chords]]
separated_mapped_chords = join_pieces(piece_mapped_chords)

# json_data = {
#     "map_to_pair": map_to_pair,
#     "pair_to_map": pair_to_map,
#     "piece_mapped_notes": piece_mapped_notes,
#     "separated_mapped_notes": separated_mapped_notes,
#     "map_to_chords":map_to_chords,
#     "chords_to_map": chords_to_map,
#     "piece_mapped_chords": piece_mapped_chords,
#     "separated_mapped_chords": separated_mapped_chords
# }


# # Open the file in write mode and write the JSON data
# with open("data.json", 'w') as json_file:
#     json.dump(json_data, json_file)

pickles = {'map_to_pair': map_to_pair, 'pair_to_map': pair_to_map, 'piece_mapped_notes': piece_mapped_notes,
           'separated_mapped_notes': separated_mapped_notes, 'map_to_chords': map_to_chords,
           'chords_to_map': chords_to_map,
           'piece_mapped_chords': piece_mapped_chords, 'separated_mapped_chords': separated_mapped_chords}
for name, dict in pickles.items():
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(dict, f)

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