from music21 import *
import os
import numpy as np
import pickle 

def extract_notes_and_chords_by_measure(mxl_file_path):
    try:
        # Load the MXL file
        score = converter.parse(mxl_file_path)

        # Initialize dictionary to store notes and chords organized by measure
        notes_by_measure = {}
        chords_by_measure = {}

        # Extract notes and chords from the score
        for part in score.parts:
            for measure_number, measure in enumerate(part.getElementsByClass('Measure')):
                measure_notes = []
                measure_chords = []
                for element in measure.recurse():
                    if isinstance(element, note.Note):
                        measure_notes.append(element)
                    elif isinstance(element, chord.Chord):
                        measure_chords.append(element)
                notes_by_measure.setdefault(measure_number, []).extend(measure_notes)
                chords_by_measure.setdefault(measure_number, []).extend(measure_chords)

        return notes_by_measure, chords_by_measure

    except converter.ConverterException as e:
        print(f"Error parsing MXL file: {e}")
        return None, None
    
# Remove extensions, inversions, and added tones from chords
def remove_extensions(chords):
    cleaned_chords = []
    for chord in chords:
        name = chord.figure
        if not name == 'N.C.':
            # Remove added tones
            add_index = name.find('add')
            if add_index != -1:
                name = name[:add_index -1]

            # Remove inversions
            inversion_index = name.find('/')
            if inversion_index != -1:
                name = name[:inversion_index]

            # Remove 6th chords
            six_index = name.find('6')
            if six_index != -1:
                name = name.replace('6', '7')
            
            # Remove 9th chords
            nine_index = name.find('9') 
            if nine_index != -1:
                name = name.replace('9', '7')

            cleaned_chords.append(name)


    return cleaned_chords

# Convert chords to integers that represent pitches
def convert_to_int(chords):
    int_pitches = []
    for chord in chords:
        h = harmony.ChordSymbol(chord)
        int_pitches.append([p.midi for p in h.pitches])
    return int_pitches

folder_path = '/home/trowan2/CS4100/cs4100-final-proj/data'
note_data = []
chord_data = []
file_names = np.array([])

# Example usage
# mxl_file_path = 'data/Afternoon_in_Paris.mxl'
for filename in os.listdir(folder_path):

    # Construct the full file path
    file_path = os.path.join(folder_path, filename)
    filename_clean = filename.replace('_', ' ').replace('.mxl', '')

    file_names = np.append(file_names, filename_clean)
    single_song_notes = []
    single_song_chords = []

    notes_by_measure, chords_by_measure = extract_notes_and_chords_by_measure(file_path)

    if notes_by_measure and chords_by_measure:
        for measure_number, notes in notes_by_measure.items():
            try:
                int_pitches = [note.pitch.midi for note in notes]
                single_song_notes.append(int_pitches)
            except:
                note_data.append([[0]])
                #print("Failed to convert") 

        for measure_number, chords in chords_by_measure.items():
            try: 
                cleaned_chords = remove_extensions(chords)
                single_song_chords.append(cleaned_chords)
            except:
                single_song_chords.append([0])
                #print("Failed to convert")
    
    note_data.append(single_song_notes)
    chord_data.append(single_song_chords)

    """
    # Printing test
    if notes_by_measure and chords_by_measure:


#     print("Notes by Measure:")
#     for measure_number, notes in notes_by_measure.items():
#         print(f"Measure {measure_number}: {notes}")
        
        #print("\nChords by Measure: ", filename_clean)
        for measure_number, chords in chords_by_measure.items():
            try: 
                cleaned_chords = remove_extensions(chords)
                int_pitches = convert_to_int(cleaned_chords)
                #print(f"Measure {measure_number}: {[chord for chord in cleaned_chords]}")

            except:
                print("Failed to convert")
    """

print(len(chord_data))
with open('note_data.pkl', 'wb') as f:
    pickle.dump(note_data, f)

with open('chord_data.pkl', 'wb') as f:
    pickle.dump(chord_data, f)

np.savetxt('song_names.txt', file_names, fmt='%s')
