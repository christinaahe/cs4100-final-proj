import pickle
import numpy as np
import copy
from tqdm import tqdm
import sys
sys.path.append(r'/Users/christinahe/Downloads/cs4100-final-proj')
from HMM import HiddenMarkovModel

FILE_PATH = r'/Users/christinahe/Downloads/cs4100-final-proj/clean_data/'

def main():
    # Load chord and note data for each piece from pickled files
    with open(FILE_PATH+"piece_mapped_chords.pkl", "rb") as infile:
        chords_by_piece = pickle.load(infile)
    with open(FILE_PATH+"piece_mapped_notes.pkl", "rb") as infile:
        notes_by_piece = pickle.load(infile)
    # Load chord and note data for all pieces from pickled files
    with open(FILE_PATH+"separated_mapped_chords.pkl", "rb") as infile:
        all_chords = pickle.load(infile)
    with open(FILE_PATH+"separated_mapped_notes.pkl", "rb") as infile:
        all_notes = pickle.load(infile)

    # Get unique chords and notes and their counts
    unique_chords = set(all_chords)
    n = len(unique_chords)
    unique_notes = set(all_notes)
    n_notes = len(unique_notes)

    # Get first and last chords for each pieve
    firsts = [elem[0] for elem in chords_by_piece]
    lasts = [elem[-1] for elem in chords_by_piece]
    # Calculate probabilities for first and last chords
    first_probs = np.array([firsts.count(i) / n for i in unique_chords])
    last_probs = np.array([lasts.count(i) / n for i in unique_chords])
    transitions = 1/n * np.ones((n, n))
    emmissions = 1/n * np.ones((n, n_notes))

    # Initialize a HMM
    model = HiddenMarkovModel(np.array(list(unique_chords)), np.array(list(unique_notes)), transitions, emmissions, first_probs, last_probs)
    # Train a model on each piece
    models = [copy.copy(model) for _ in notes_by_piece]
    trained_models = []
    for i in tqdm(range(len(models))):
        trained_models.append(models[i].train(notes_by_piece[i]))

    # Calculate average emissions and transitions from trained models
    emmissions = np.mean([elem.E for elem in models], axis=0)
    transitions = np.mean([elem.T for elem in models], axis=0)

    # Print and save the emission and transition probabilities
    print(emmissions)
    print(transitions)
    np.savetxt("transitions.csv", transitions, delimiter=",", fmt='%.20f')
    np.savetxt("emissions.csv", emmissions, delimiter=",", fmt='%.20f')


if __name__ == "__main__":
    main()
