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
    # transitions = 1/n * np.ones((n, n))
    # emmissions = 1/n * np.ones((n, n_notes))

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

    with open ("transitions.csv", "r") as infile:
        transitions = np.genfromtxt(infile, delimiter=",")
    with open ("emissions.csv", "r") as infile:
        emmissions = np.genfromtxt(infile, delimiter=",")

    # print(np.where(emmissions[1, :] == 0))
    # print(emmissions.shape)
    test_notes = notes_by_piece[10][:10]
    # print(test_notes)
    # with open("pair_to_map.pkl", "rb") as infile:
    #     pair_to_map = pickle.load(infile)
    print(test_notes)

    model = HiddenMarkovModel(np.array(list(unique_chords)), np.array(list(unique_notes)), transitions, emmissions, first_probs, last_probs)

    with open("map_to_chords.pkl", "rb") as infile:
        map_to_chords = pickle.load(infile)
    
    print(map_to_chords)
    
    
    seq = model.predict_algorithm(test_notes)
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