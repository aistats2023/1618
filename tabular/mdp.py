import numpy as np
import numpy.linalg as npl
import numpy.random as npr

##########################################################

class tabular():
    # represents the MRP

    def __init__(self, Pmat, reward, gamma):
        # initializes the MRP
        # Pmat: numpy matrix of size dim x dim of positive entries
        # reward: numpy array float (dim), reward vector
        # gamma: float, discount factor

        self.Pmat = Pmat 
        self.reward = reward
        self.gamma = gamma
        self.dim = Pmat.shape[0]

        self.normalize_transmat()
        self.compute_stationary()
        self.compute_value()

    def normalize_transmat(self):
        # normalize the rows of pmat

        sums = np.sum(self.Pmat, axis = 1)
        for i in range(self.dim):
            self.Pmat[i, :] /= sums[i]


    def compute_stationary(self, num_iter = 10000):
        # computes the stationary distribution of MC given by pmat
        # we're gonna brute force this since we don't have time

        stationary = npr.rand(self.dim)
        stationary = stationary / stationary.sum()

        for i in range(num_iter):
            stationary = self.Pmat.T @ stationary

        self.stationary = stationary


    def collect_samples(self, num_samples):
        # collects samples into an array format
        # num_samples: int representing number of samples to return
        #
        # returns
        # curr_state: numpy int array of dimension num_samples representing
        #               the starting state in a transition
        # rew: numpy array of size num_samples representing rewards in a transition
        # next_state: numpy int array of dimension num_samples representing
        #                finishing state in a transition

        curr_states = np.zeros(num_samples)
        next_states = np.zeros(num_samples)
        rew = np.zeros(num_samples)

        for i in range(num_samples):
            state = npr.choice(self.dim, p = self.stationary)

            curr_states[i] = state
            rew[i] = self.reward[state]
            next_states[i] = npr.choice(self.dim, p = self.Pmat[state, :])

        return curr_states, rew, next_states


    def compute_value(self):
        # compute the value function
        self.value = npl.solve(np.eye(self.dim) - self.gamma * self.Pmat, self.reward)


    def evaluate_value(self, est_value):
        # computes the L2-mu norm between the given value functions
        # est_value: numpy float matrix (dim) representing value function

        diff = est_value - self.value
        return np.sqrt((self.stationary * diff * diff).sum())