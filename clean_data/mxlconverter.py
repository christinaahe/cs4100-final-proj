import music21
from music21 import *
import os
import numpy as np
import pickle 

def extract_notes_and_chords_by_measure(mxl_file_path):
    """
    Gets notes and chords from a MusicXML (MXL) file organized by measure.

    Parameters:
    - mxl_file_path (str): The file path of the MXL file to extract notes and chords from.

    Returns:
    - notes_by_measure (dict): A dictionary where keys are measure numbers (int) and values are lists of notes and rests
                               (music21.note.Note or music21.note.Rest objects) occurring in that measure.
    - chords_by_measure (dict): A dictionary where keys are measure numbers (int) and values are lists of chords
                                (music21.chord.Chord objects) occurring in that measure.
    """
    try:
        # Load the MXL file
        score = converter.parse(mxl_file_path)

        # Initialize dictionary to store notes and chords organized by measure
        notes_by_measure = {}
        chords_by_measure = {}
    # Extract notes, rests, and chords from the score
        for part in score.parts:
            # iterate through each measure
            for measure_number, measure in enumerate(part.getElementsByClass('Measure')):
                measure_notes = []
                measure_chords = []
                for element in measure.recurse():
                    # add notes and rests to measure_notes, and chords to measure_chords
                    if isinstance(element, music21.note.Note):
                        measure_notes.append(element)
                    elif isinstance(element, music21.note.Rest):
                        measure_notes.append(element)
                    elif isinstance(element, music21.chord.Chord):
                        measure_chords.append(element)
                # add measure notes, rests, and chords to dictionaries
                notes_by_measure.setdefault(measure_number, []).extend(measure_notes)
                chords_by_measure.setdefault(measure_number, []).extend(measure_chords)

        return notes_by_measure, chords_by_measure

    except converter.ConverterException as e:
        print(f"Error parsing MXL file: {e}")
        return None, None
    
def remove_extensions(chords):
    """
    Remove extensions, inversions, and added tones from chords.

    Parameters:
    - chords (list): A list of music21.chord.Chord objects representing chords to be cleaned.

    Returns:
    - cleaned_chords (list): A list of cleaned chord names (str) without extensions, inversions, and added tones.
    """
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

            # chord_length = max([n.quarterLength for n in chord.notes])
            cleaned_chords.append(name)


    return cleaned_chords

# Convert chords to integers that represent pitches
def convert_to_int(chords):
    int_pitches = []
    for chord in chords:
        h = harmony.ChordSymbol(chord)
        int_pitches.append([p.midi for p in h.pitches])
    return int_pitches


def main():
    folder_path = r'/Users/christinahe/Documents/spring2024/cs4100/project/cs4100-final-proj/data'
    note_data = []
    chord_data = []
    file_names = np.array([])

    # Iterate through each file in the data folder
    for filename in os.listdir(folder_path):

        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        # Get the name of each piece and append to array
        filename_clean = filename.replace('_', ' ').replace('.mxl', '')
        file_names = np.append(file_names, filename_clean)

        # Initialize lists to store notes and chords for each song
        single_song_notes = []
        single_song_chords = []

        # Extract notes and chords organized by measure from the MusicXML file
        notes_by_measure, chords_by_measure = extract_notes_and_chords_by_measure(file_path)

        if notes_by_measure and chords_by_measure:
            # Process notes for each measure
            for _, notes in notes_by_measure.items():
                try:
                    # Convert note pitches to MIDI representation and store with duration
                    int_pitches = []
                    for note in notes:
                        if note.isRest:
                            int_pitches.append((-1, note.duration.quarterLength))
                        elif note.isNote:
                            int_pitches.append((note.pitch.midi, note.duration.quarterLength))

                    single_song_notes.append(int_pitches)
                except:
                    # If conversion fails, append default 0
                    note_data.append([[0]])
                    #print("Failed to convert") 
            
            # Process chords for each measure
            for _, chords in chords_by_measure.items():
                try:
                    # Remove extensions, inversions, and added tones from chords
                    cleaned_chords = remove_extensions(chords)
                    single_song_chords.append(cleaned_chords)
                except:
                     # If conversion fails, append default 0
                    single_song_chords.append([0])
                    #print("Failed to convert")
        
        # Append the notes and chords for the current song to the overall data lists
        note_data.append(single_song_notes)
        chord_data.append(single_song_chords)

    # Pickle and store data
    # with open('note_rest_data.pkl', 'wb') as f:
    #     pickle.dump(note_data, f)

    # with open('chord_data.pkl', 'wb') as f:
    #     pickle.dump(chord_data, f)

    # np.savetxt('song_names.txt', file_names)

if __name__ == '__main__':
    main()
