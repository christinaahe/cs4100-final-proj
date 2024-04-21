import os
import json
import time
from datetime import datetime


from masterpiece import Masterpiece
from librosa import note_to_midi
import predict_chords_utils as pcu

# global variable for all notes, with enharmonics (different names for the same note, like C# and Db)
NOTES_WITH_ENHARMONICS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "Fb", "E#", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B", "Cb", "B#"]

def generate_midi():
    """Generates a MIDI file based on user input."""
    dtime = datetime.now()
    ans_time = time.mktime(dtime.timetuple())
    params_file = open("./track_generation_files/song_settings.json", "r")
    params = json.load(params_file)
    params_file.close()

    # write notes into rules.json or read from rules.json
    input_notes = input("Do you want to input notes? [y/n]: ")
    if input_notes == "n":
        # read notes from rules.json
        with open("./track_generation_files/rules.json", "r") as f:
            rules = json.load(f)
            notes = rules["notes"]
            durations = rules["rhythm"][0]

    if input_notes == "y":
        # write notes into rules.json
        notes = []
        durations = []
        while True:
            print("Welcome to the note input interface!")
            note = input("Enter a note [q to end]: ")
            if note == 'q':
                break
            duration = input("Enter a duration for that note (as a float, how many beats you want that note to be) [q to end]: ")

            if duration == "q":
                break
            os.system("clear") # clear the terminal screen so it doesn't get crowded

            try:
                if note not in NOTES_WITH_ENHARMONICS:
                    raise ValueError
                notes.append(note_to_midi(f'{note}4')) # convert note to MIDI note using librosa, the octave is 4 (the octave used in our training)
            except ValueError:
                print("Invalid input for note. Please try again.") # if the note is not in the list of notes, it's invalid
                continue
            try:
                durations.append(float(duration))
            except ValueError:
                print("Invalid input for duration. Please try again.") # if the duration is not a float, it's invalid, so we ask the user to try again
                continue

        # check if the number of notes and durations match
        if len(notes) != len(durations):
            print("The number of notes and durations do not match. Please try again.")
            exit(1)

        # check if the user entered any notes
        if len(notes) == 0:
            print("You did not enter any notes. Please try again.")
            exit(1)

    # get chords and chord durations from notes and durations using the HMM #
    midi_chord_seq = pcu.get_chord_sequence(notes, durations)


    chords = midi_chord_seq["chords"]
    chord_durations = midi_chord_seq["chord_durations"]

    # get chord notes from chords using chord_to_notes.json
    with open("./track_generation_files/chord_to_notes.json", "r") as f:
        chord_to_notes = json.load(f)

    chord_notes = []
    for chord in chords:
        if chord in chord_to_notes:
            chord_notes.append(chord_to_notes[chord]["midi_notes"])

    with open("./track_generation_files/rules.json", "r") as f:
        rules = json.load(f)

    rules["notes"] = notes
    rules["rhythm"] = [durations] # wrap durations in a list to match the format of the rhythm in rules.json
    rules["chord_rhythm"] = chord_durations
    rules["seq_chord"] = chord_notes

    with open("./track_generation_files/rules.json", "w") as f:
        json.dump(rules, f)

    # create the masterpiece class and generate the MIDI file
    my_masterpiece = Masterpiece(
        rules_path="./track_generation_files/rules.json",
        length=params["length"],
        tempo=params["tempo"])

    subfolder = "track_outputs"

    # create the output folder if it doesn't exist
    if not os.path.isdir(subfolder):
        os.mkdir(subfolder)

    # create the MIDI file
    my_masterpiece.create_midi_file("{folder}/midi_{suffix}.mid".format(
        folder=subfolder,
        suffix=ans_time))

if __name__ == "__main__":
    generate_midi()