import copy

import numpy as np


class HiddenMarkovModel:
    def __init__(
        self,
        true_states,
        observable_states,
        transitions,
        emissions,
        start_probs,
        end_probs,
    ):
        self.true_states = true_states
        self.observable_states = observable_states
        self.T = transitions
        self.E = emissions
        self.start_probs = start_probs
        self.end_probs = end_probs
        # E: [state, observation]
        # T: [state, next_state]
        # B: [state, observation]
        # F: [state, observation]
        # eta: [state, observation]
        # xi: [state, observation, next_state]

    def forward(self, sequence):
        forward_vals = np.zeros((len(self.true_states), len(sequence)))
        for i in range(len(sequence)):
            state_idx = sequence[i]
            for j in range(len(self.true_states)):
                # case for start of sequence
                if i == 0:
                    forward_vals[j, i] = self.start_probs[j] * self.E[j, self.observable_states[state_idx]]
                else:
                    vals = np.array(
                        [
                            forward_vals[k, i - 1] * self.E[j, self.observable_states[state_idx]] * self.T[k, j]
                            for k in range(len(self.true_states))
                        ]
                    )
                    forward_vals[j, i] = np.sum(vals)
        end = np.multiply(forward_vals[:, -1], self.end_probs)
        end_val = np.sum(end)
        return forward_vals, end_val

    def backward(self, sequence):
        backward_vals = np.zeros((len(self.true_states), len(sequence)))
        #for i in range(1, len(sequence) + 1, -1):
        for i in range(1, len(sequence) + 1):
            for j in range(len(self.true_states)):
                # case for end of sequence
                if i == 1:
                    backward_vals[j, -i] = self.end_probs[j]
                else:
                    vals = np.array(
                        [
                            backward_vals[k, -i + 1] * self.E[k, self.observable_states[sequence[-i + 1]]] * self.T[k, j]
                            for k in range(len(self.true_states))
                        ]
                    )
                    backward_vals[j, -i] = np.sum(vals)
        start_state = [
            backward_vals[i, 0] * self.E[i, self.observable_states[sequence[0]]]
            for i in range(len(self.true_states))
        ]
        start_state = np.multiply(start_state, self.start_probs)
        start_val = np.sum(start_state)
        return backward_vals, start_val

    def eta(self, forward_probs, backward_probs, forward_val, sequence):
        eta_probs = np.zeros((len(self.true_states), len(sequence)))
        for i in range(len(sequence)):
            for j in range(len(self.true_states)):
                eta_probs[j, i] = (
                    forward_probs[j, i] * backward_probs[j, i]
                ) / forward_val
        return eta_probs

    def xi(self, forward_probs, backward_probs, forward_val, sequence):
        xi_probs = np.zeros(
            (len(self.true_states), len(sequence) - 1, len(self.true_states)))
        for i in range(len(sequence) - 1):
            for j in range(len(self.true_states)):
                for k in range(len(self.true_states)):
                    xi_probs[j, i, k] = (
                        forward_probs[j, i]
                        * backward_probs[k, i + 1]
                        * self.E[k, self.observable_states[sequence[i + 1]]] * self.T[j, k]
                    ) / forward_val
        return xi_probs

    def train(self, sequence):
        forward_probs, forward_val = self.forward(sequence)
        for _ in range(1000):
            # forward_probs, forward_val = self.forward(sequence)
            backward_probs, backward_val = self.backward(sequence)
            eta_probs = self.eta(forward_probs, backward_probs, forward_val, sequence)
            xi_probs = self.xi(forward_probs, backward_probs, forward_val, sequence)

            # recalculate transitions and emissions
            E = np.zeros(self.E.shape)
            T = np.zeros(self.T.shape)
            for i in range(len(self.true_states)):
                for j in range(len(self.true_states)):
                    for k in range(len(sequence) - 1):
                        # sum xi for numerator
                        T[i, j] += xi_probs[i, k, j]
                    # sum xi for denominator
                    #xi_sum = np.array([eta_probs[i_x, k_x] for i_x in range(len(self.true_states)) for k_x in range(len(sequence))])
                    xi_sum = np.array([xi_probs[j, k_x, i_x] for k_x in range(len(sequence) - 1) for i_x in range(len(self.true_states))])
                    xi_sum = np.sum(xi_sum)
                    if xi_sum != 0:
                        T[i, j] /= xi_sum
                    else:
                        T[i, j] = 0
            for i in range(len(self.true_states)):
                for j in range(len(sequence)):
                    # get eta sums
                    idxs = [k for k in range(len(sequence)) if sequence[k] == self.observable_states[j]]
                    eta_sum_numerator = np.sum(eta_probs[i, idxs])
                    eta_sum_denominator = np.sum(eta_probs[i, :])
                    if eta_sum_denominator != 0:
                        E[i, j] = eta_sum_numerator / eta_sum_denominator
                    else:
                        E[i, j] = 0
            self.E = E
            self.T = T
            print("-----------------------")
            print("Update Iteration Complete")

            temp_forward_val = copy.copy(forward_val)
            forward_probs, forward_val = self.forward(sequence)
            print(temp_forward_val)
            print(forward_val)
            diff = np.abs(forward_val - temp_forward_val)
            if diff <= 0.0000001:
                break


    def predict_algorithm(self, obs):
        # Step 2: Initialize Variables
        predict_table = [
            [0.0 for _ in range(len(self.true_states))] for _ in range(len(obs))
        ]
        backpointer = [
            [0 for _ in range(len(self.true_states))] for _ in range(len(obs))
        ]

        # Step 3: Calculate Probabilities
        for t in range(len(obs)):
            for s in range(len(self.true_states)):
                if t == 0:
                    predict_table[t][s] = self.start_probs[s] * self.E[s][obs[t]]
                else:
                    max_prob = max(
                        predict_table[t - 1][prev_s] * self.T[prev_s][s]
                        for prev_s in range(len(self.true_states))
                    )
                    predict_table[t][s] = max_prob * self.E[s][obs[t]]
                    backpointer[t][s] = max(
                        range(len(self.true_states)),
                        key=lambda prev_s: predict_table[t - 1][prev_s]
                        * self.T[prev_s][s],
                    )

        # Step 4: Traceback and Find Best Path
        best_path_prob = max(predict_table[-1])
        best_path_pointer = max(
            range(len(self.true_states)), key=lambda s: predict_table[-1][s]
        )
        best_path = [best_path_pointer]
        for t in range(len(obs) - 1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])

        # Step 5: Return Best Path
        return best_path
    


