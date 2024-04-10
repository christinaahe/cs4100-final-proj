import pickle
import numpy as np
from HMM import HiddenMarkovModel
import copy
from tqdm import tqdm

def main():
    with open("piece_mapped_chords.pkl", "rb") as infile:
        chords_by_piece = pickle.load(infile)
    with open("piece_mapped_notes.pkl", "rb") as infile:
        notes_by_piece = pickle.load(infile)
    all_chords = []
    for piece in chords_by_piece:
        all_chords += piece
    unique_chords = set(all_chords)
    n = len(unique_chords)
    all_notes = []
    for piece in notes_by_piece:
        all_notes += piece
    unique_notes = set(all_notes)
    n_notes = len(unique_notes)
    firsts = [elem[0] for elem in chords_by_piece]
    lasts = [elem[-1] for elem in chords_by_piece]
    first_probs = np.array([firsts.count(i) / n for i in unique_chords])
    last_probs = np.array([lasts.count(i) / n for i in unique_chords])
    transitions = 1/n * np.ones((n, n))
    emmissions = 1/n * np.ones((n, n_notes))

    model = HiddenMarkovModel(np.array(list(unique_chords)), np.array(list(unique_notes)), transitions, emmissions, first_probs, last_probs)
    sequence = notes_by_piece[5]
    models = [copy.copy(model) for _ in notes_by_piece]
    trained_models = []
    for i in tqdm(range(len(models))):
    #for i in range(2):

        trained_models.append(models[i].train(notes_by_piece[i]))
    emmissions = np.maximum([elem.E for elem in models], axis=0)
    transitions = np.mean([elem.T for elem in models], axis=0)
    print(emmissions)
    print(transitions)

    """
    model.train(sequence)
    print(model.E)
    print(model.T)
    print(np.count_nonzero(model.E))
    print(np.count_nonzero(model.T))
    """



if __name__ == "__main__":
    main()