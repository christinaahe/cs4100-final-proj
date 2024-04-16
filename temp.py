import copy
import pickle

import numpy as np
from tqdm import tqdm

from HMM import HiddenMarkovModel


def main():
    with open("piece_mapped_chords.pkl", "rb") as infile:
        chords_by_piece = pickle.load(infile)
    with open("piece_mapped_notes.pkl", "rb") as infile:
        notes_by_piece = pickle.load(infile)
    all_chords = []
    for piece in chords_by_piece:
        all_chords += piece
    unique_chords = set(all_chords)
    print(unique_chords)
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

    # model = HiddenMarkovModel(np.array(list(unique_chords)), np.array(list(unique_notes)), transitions, emmissions, first_probs, last_probs)
    # sequence = notes_by_piece[5]
    # models = [copy.copy(model) for _ in notes_by_piece]
    # trained_models = []
    # for i in tqdm(range(len(models))):
    # #for i in range(2):
    #     trained_models.append(models[i].train(notes_by_piece[i]))
    # emmissions = np.mean([elem.E for elem in models], axis=0)
    # transitions = np.mean([elem.T for elem in models], axis=0)
    # print(np.count_nonzero(emmissions == 0))
    # print(np.count_nonzero(transitions == 0))

    # np.savetxt("transitions.csv", transitions, delimiter=",", fmt='%.150f')
    # np.savetxt("emissions.csv", emmissions, delimiter=",", fmt='%.150f')

    # with open("transitions.csv", "r") as infile:
    #     transitions = np.genfromtxt(infile, delimiter=",")
    # with open("emissions.csv", "r") as infile:
    #     emmissions = np.genfromtxt(infile, delimiter=",")

    # print(np.where(emmissions[1, :] == 0))
    # print(emmissions.shape)
    test_notes = notes_by_piece[14][:5]
    # print(test_notes)
    # with open("pair_to_map.pkl", "rb") as infile:
    #     pair_to_map = pickle.load(infile)
    print(test_notes)


    print(len(unique_chords))
    print(transitions.shape)
    print(emmissions.shape)

    model = HiddenMarkovModel(
        np.array(list(unique_chords)),
        np.array(list(unique_notes)),
        transitions,
        emmissions,
        first_probs,
        last_probs,
    )

    with open("map_to_chords.pkl", "rb") as infile:
        map_to_chords = pickle.load(infile)

    print(map_to_chords)

    first_probs = np.array([firsts.count(i) / n for i in unique_chords])
    first_probs /= first_probs.sum()

    seq = model.predict_algorithm(test_notes, first_probs)
    print(seq)


if __name__ == "__main__":
    main()

    """40,  44,  57,  59,  62,  63,  64,  65,  66,  70,  71,  73,  74,
        75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
        88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100,
       101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
       114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
       127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138
    """

"""def predict_algorithm(self, y, pi):
        B = self.E
        A = self.T
        N = B.shape[0]

        x_seq = np.zeros(len(y), dtype=int)

        x_seq[0] = np.random.choice(N, p=pi)

        V = B[:, y[0]] * pi

        # forward to compute a LIKELY value function V
        for i, y_ in enumerate(y[1:], start=1):
            print(B[:, y_].shape, A.shape, V.shape)  # Add this line to check the shapes
            print(x_seq)
            _V = np.tile(B[:, y_], reps=[N, 1]).T * A.T * np.tile(V, reps=[N, 1])
            _V /= (np.sum(_V, axis=1, keepdims=True) + 1e-10)  # Add a small constant to avoid division by zero

            # Normalize each row of _V to ensure that the probabilities sum to 1
            _V = _V / _V.sum(axis=1, keepdims=True)

            # 2/3 of the time, choose the optimal path
            if np.random.choice([True, False], p=[2/5, 3/5]):
                x_seq[i] = np.unravel_index(np.argmax(_V), _V.shape)[1]
                print("Optimal")
                print(x_seq[i])
            else:
                x_seq[i] = np.random.choice(range(len(_V[x_seq[i-1]])), p=_V[x_seq[i-1]])
                print("Random")
                print(x_seq[i])

            print(_V.shape)
            V = _V[np.arange(N), x_seq[i]]  # update V

        x_T = np.argmax(V)

        # backward to fetch optimal sequence
        x_seq_opt, i = np.zeros(len(x_seq)), len(x_seq) - 1
        prev_ind = x_T
        while i >= 0:
            x_seq_opt[i] = prev_ind
            i -= 1
            prev_ind = x_seq[i]
        return x_seq_opt"""