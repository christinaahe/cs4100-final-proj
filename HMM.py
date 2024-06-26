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
        """
        calculates the joint probability of observed data up to time k and the state at time k
        p(obsv 0:k, state k)
        """
        forward_vals = np.zeros((len(self.true_states), len(sequence)))
        for i in range(len(sequence)):
            state_idx = sequence[i]
            for j in range(len(self.true_states)):
                # case for start of sequence (need this bc formula is recursive and must have start value)
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
        """
        calculates the conditional probabilities of the observed data from time k+1 given the state at time k
        p(obsv k+1: | state k)
        """
        backward_vals = np.zeros((len(self.true_states), len(sequence)))
        for i in range(1, len(sequence) + 1):
            for j in range(len(self.true_states)):
                # case for end of sequence (this is recursive and needs a start value)
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
        """
        calculates the probability distribution of states at each time k given the complete observation sequence
        """
        eta_probs = np.zeros((len(self.true_states), len(sequence)))
        for i in range(len(sequence)):
            for j in range(len(self.true_states)):
                eta_probs[j, i] = (
                    forward_probs[j, i] * backward_probs[j, i]
                ) / forward_val
        return eta_probs

    def xi(self, forward_probs, backward_probs, forward_val, sequence):
        """
        calculates the joint probabilities of all consecutive state pairs given the complete observation sequence
        """
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
        """
        tunes the transition and emission matrices so the model is maximally like the observed data
        """
        forward_probs, forward_val = self.forward(sequence)
        for _ in range(iterations):
            backward_probs, backward_val = self.backward(sequence)
            eta_probs = self.eta(forward_probs, backward_probs, forward_val, sequence)
            xi_probs = self.xi(forward_probs, backward_probs, forward_val, sequence)

            # recalculate transitions and emissions
            E = np.zeros(self.E.shape)
            T = np.zeros(self.T.shape)
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
            print("-----------------------")
            print("Update Iteration Complete")
            temp_forward_val = copy.copy(forward_val)
            forward_probs, forward_val = self.forward(sequence)
            diff = np.abs(forward_val - temp_forward_val)
            if diff <= 0.0000001:
                break
        return xi_probs, eta_probs


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
        return np.array(best_path)
    
    def predict_chords(self, observations, top_n=3):
        # Initialize variables
        n_states = len(self.true_states)
        n_observations = len(observations)
        delta = np.zeros((n_states, n_observations))
        psi = np.zeros((n_states, n_observations), dtype=int)

        # Initialization step
        delta[:, 0] = self.start_probs * self.E[:, observations[0]]

        # Recursion step
        for t in range(1, n_observations):
            for j in range(n_states):
                # Calculate probabilities for top-N states
                top_n_probs = delta[:, t-1] * self.T[:, j] * self.E[j, observations[t]]
                # Select top-N states
                top_n_states = np.argsort(top_n_probs)[::-1][:top_n]
                # Randomly select one state from the top-N states
                selected_state = np.random.choice(top_n_states)
                delta[j, t] = top_n_probs[selected_state]
                psi[j, t] = selected_state

        # Termination step
        p_max = np.max(delta[:, -1])
        q_star = np.argmax(delta[:, -1])

        # Backtracking
        states = [q_star]
        for t in range(n_observations - 1, 0, -1):
            states.insert(0, psi[states[0], t])

        # Map state indices to true states (chords)
        predicted_chords = [self.true_states[state] for state in states]

        return predicted_chords