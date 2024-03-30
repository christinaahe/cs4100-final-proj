from music21 import *
import os

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

folder_path = 'data'

for filename in os.listdir(folder_path):
    # Construct the full file path
    file_path = os.path.join(folder_path, filename)
# # Example usage
# mxl_file_path = 'data/Afternoon_in_Paris.mxl'
    print(file_path)
    notes_by_measure, chords_by_measure = extract_notes_and_chords_by_measure(file_path)
    print(notes_by_measure, chords_by_measure)

# if notes_by_measure and chords_by_measure:
#     print("Notes by Measure:")
#     for measure_number, notes in notes_by_measure.items():
#         print(f"Measure {measure_number}: {notes}")
    
#     print("\nChords by Measure:")
#     for measure_number, chords in chords_by_measure.items():
#         print(f"Measure {measure_number}: {chords}")