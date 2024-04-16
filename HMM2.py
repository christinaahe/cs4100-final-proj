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
                        [forward_vals[k, i - 1] * self.E[j, self.observable_states[state_idx]] * self.T[k, j]
                            for k in range(len(self.true_states))])
                    forward_vals[j, i] = np.sum(vals)
        end = np.multiply(forward_vals[:, -1], self.end_probs)
        end_val = np.sum(end)
        return forward_vals, end_val

    def backward(self, sequence):
        backward_vals = np.zeros((len(self.true_states), len(sequence)))
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

    def train(self, sequence, iterations=1000, end=False):
        forward_probs, forward_val = self.forward(sequence)
        for _ in range(iterations):
            # forward_probs, forward_val = self.forward(sequence)
            backward_probs, backward_val = self.backward(sequence)
            eta_probs = self.eta(forward_probs, backward_probs, forward_val, sequence)
            xi_probs = self.xi(forward_probs, backward_probs, forward_val, sequence)

            # recalculate transitions and emissions
            E = np.zeros(self.E.shape)
            T = np.zeros(self.T.shape)
            """
            for i in range(len(self.true_states)):
                for j in range(len(self.true_states)):
                    ###
                    for k in range(len(sequence) - 1):
                        # sum xi eta for numerator
                        T[i, j] += xi_probs[i, k, j]
                    ###
                    T[i, j] = np.sum(xi_probs[i, :, j])
                    # sum xi eta for denominator numerator
                    #xi_sum = np.array([xi_probs[j, k_x, i_x] for k_x in range(len(sequence) - 1) for i_x in range(len(self.true_states))])
                    #xi_sum = np.sum(xi_sum)
                    eta_sum = np.sum(eta_probs[i])
                    #if xi_sum != 0:
                    #    T[i, j] /= xi_sum
                    if eta_sum != 0:
                        T[i, j] /= eta_sum
                    else:
                        T[i, j] = 0
            """
            print("---------------")
            # for i in self.true_states:
            #     state_prob = sum(eta_probs[i])
            #     T[0, i] = eta_probs[i, 0]
            #     if state_prob == 0:
            #         T[i, :] = 0
            #         continue
            #     T[i, -1] = eta_probs[i, -1] / state_prob
            #     for j in range(len(self.true_states) - 1):
            #         T[i, j] = sum(xi_probs[i, :, j]) / state_prob
            for i in self.true_states:
                state_prob = sum(eta_probs[i])
                T[0, i] = eta_probs[i, 0]
                val = eta_probs[i, -1]
                if val == 0:
                    T[i, -1] = 0
                elif state_prob == 0:
                    T[i, -1] = 1
                else:
                    T[i, -1] = val / state_prob
                for j in range(len(self.true_states) - 1):
                    val = sum(xi_probs[i, :, j])
                    if val == 0:
                        T[i, j] = 0
                    elif state_prob == 0:
                        T[i, j] = 1
                    else:
                        T[i, j] = val / state_prob
            for i in range(len(self.true_states)):
                for j in range(len(self.observable_states)):
                    # get eta sums
                    idxs = [idx for idx, val in enumerate(sequence) if val == self.observable_states[j]]

                    eta_sum_numerator = np.sum(eta_probs[i, idxs])
                    eta_sum_denominator = np.sum(eta_probs[i, :])
                    if eta_sum_denominator != 0:
                        E[i, j] = eta_sum_numerator / eta_sum_denominator
                    else:
                        E[i, j] = 0
            if end:
                T[self.true_states[-1], self.true_states[-1]] = 1
            self.E = E
            self.T = T
            self.start_probs = self.T[0]
            self.end_probs = self.T[:, -1]
            #print(eta_probs)
            #print("-----------------------")
            #print("Update Iteration Complete")
            print(forward_val)
            print(self.T)
            temp_forward_val = copy.copy(forward_val)
            forward_probs, forward_val = self.forward(sequence)
            diff = np.abs(forward_val - temp_forward_val)
            #if diff <= 0.0000001:
            #    break
        return xi_probs, eta_probs

    def predict_algorithm(self, y, pi):

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
            # print("First value of _V at step", i, ":", _V[0])
            _V /= (np.sum(_V, axis=1, keepdims=True) + 1e-10)  # Add a small constant to avoid division by zero

            # Normalize each row of _V to ensure that the probabilities sum to 1
            _V = _V / _V.sum(axis=1, keepdims=True)

            # 2/3 of the time, choose the optimal path
            if np.random.choice([True, False], p=[2/3, 1/3]):
                x_seq[i] = np.unravel_index(np.argmax(_V), _V.shape)[1]
                print("Optimal")
                print(x_seq[i])
            else:
                print("Random")
                sorted_indices = np.argsort(_V[x_seq[i-1]])[::-1]
                x_seq[i] = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
                print(x_seq[i])

            print(_V.shape)
            V = _V[np.arange(N), x_seq[i]]  # update V

        x_T = np.argmax(V)

        # backward to fetch optimal sequence
        x_seq_opt, i = np.zeros(len(x_seq)), len(x_seq) - 1
        prev_ind = x_T
        while i >= 0:
            x_seq_opt[i] = prev_ind
            if i > 0:  # Only update prev_ind if there are more elements
                prev_ind = x_seq[i-1]
            i -= 1
        return x_seq_opt


