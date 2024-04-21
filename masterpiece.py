# adapted from https://github.com/Jobsecond/random-midi

import json

from midiutil.MidiFile import MIDIFile


class Masterpiece(object):
    """Object that generates a MIDI file based on a set of rules.
    """

    def __init__(self, rules_path="./track_generation_files/rules.json", length=4, tempo=90):
        self.rules_path = rules_path
        self.length = length
        self.tempo = tempo

        rules_file = open(rules_path, "r")
        rules = json.load(rules_file)
        rules_file.close()
        self.rhythm = rules["rhythm"]
        self.seq_chord = rules["seq_chord"]
        self.chord_rhythm = rules["chord_rhythm"]
        self.velocity = rules["velocity"]
        self.notes = rules["notes"]

        self.MyMIDI = MIDIFile(3)
        self.current_track_number = 0

    def create_melody_sequence(self):
        """Create a sequence of notes and durations for the melody.

        Returns:
            list : list of tuples containing the note and duration
        """
        seq_melody = []
        for i in range(self.length):
            for phrase in self.rhythm:
                for note, duration in zip(self.notes, phrase):
                    seq_melody.append((note, duration))
        return seq_melody

    def create_melody_track(self):
        """Create the melody track in the MIDI file.
        """
        seq_melody = self.create_melody_sequence()

        self.MyMIDI.addTrackName(
            track=self.current_track_number,
            time=0, trackName="piano")
        self.MyMIDI.addTempo(
            track=self.current_track_number,
            time=0, tempo=self.tempo)
        self.MyMIDI.addProgramChange(
            tracknum=self.current_track_number,
            channel=0, time=0, program=0)

        pos = 0
        for pitch, duration in seq_melody:
            relative_pos = pos - int(pos / 4) * 4
            if 0 <= relative_pos < 1:
                vol = self.velocity["strong"]
            elif 2 <= relative_pos < 3:
                vol = self.velocity["intermediate"]
            else:
                vol = self.velocity["weak"]
            self.MyMIDI.addNote(
                track=self.current_track_number,
                channel=0, pitch=pitch, time=pos, duration=duration, volume=vol)
            if relative_pos in [0, 2]:
                self.MyMIDI.addControllerEvent(
                    track=self.current_track_number,
                    channel=0, time=pos, controller_number=64, parameter=127)
                self.MyMIDI.addControllerEvent(
                    track=self.current_track_number,
                    channel=0, time=pos + 1.96875, controller_number=64, parameter=0)
            pos += duration
        self.current_track_number += 1

    def create_chord_track(self):
        """Create the chord track in the MIDI file.
        """
        self.MyMIDI.addTrackName(
            track=self.current_track_number,
            time=0, trackName="chords")
        self.MyMIDI.addTempo(
            track=self.current_track_number,
            time=0, tempo=self.tempo)
        self.MyMIDI.addProgramChange(
            tracknum=self.current_track_number,
            channel=0, time=0, program=0)

        pos = 0
        for chord, rhythm in zip(self.seq_chord, self.chord_rhythm):
            for pitch in chord:
                self.MyMIDI.addControllerEvent(
                    track=self.current_track_number,
                    channel=0, time=pos, controller_number=64, parameter=127)
                self.MyMIDI.addControllerEvent(
                    track=self.current_track_number,
                    channel=0, time=pos + rhythm - 0.03125, controller_number=64, parameter=0)
                self.MyMIDI.addNote(
                    track=self.current_track_number,
                    channel=0, pitch=pitch, time=pos, duration=rhythm, volume=76)
            pos += rhythm
        self.current_track_number += 1

    def create_midi_file(self, filename, melody=True, chord=True):

        """Create the MIDI file based on the rules."""
        if melody:
            self.create_melody_track()
        if chord:
            self.create_chord_track()
        with open(filename, "wb") as midi_file:
            self.MyMIDI.writeFile(midi_file)
