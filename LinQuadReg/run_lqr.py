# file runs policy evaluation for infinite horizon LQR with discounting
# implements an LQR system that is comptabile with lstd_boost

import numpy as np
import pandas as pd
import numpy.linalg as npla
import scipy.linalg as scila
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from lstd_boost import *
from fitted_value import *

##########################################################

class LQR():
    # object that represents an LQR instance with a policy

    def __init__(self, A, B, K, noise_cov, Q, R, gamma):
        # initializes the LQR system
        # A: numpy matrix of dimension d x d representing system dynamics
        # B: numpy matrix of dimension d x m representing control dynamics
        # K: numpy matrix of dimension m x d representing policy
        # noise_sd: numpy matrix of dimension dxd for the covariance of the noise
        # Q: numpy matrix of dimension d x d representing state cost
        # R: numpy matrix of dimension m x m representing action cost

        # sets parameters
        self.A = A
        self.B = B
        self.K = K
        self.noise_cov = noise_cov
        self.Q = Q
        self.R = R
        self.gamma = gamma

        self.state_dim = np.shape(A)[0]
        self.action_dim = np.shape(B)[1]

        # initializes LQR at 0
        self.state = np.zeros(self.state_dim)

        # computes and saves transition and cost matrices
        self.trans_mat = A + B @ K
        self.cost_mat = Q + K.T @ R @ K

        # checks stabilizing
        eigvals = npla.eigvals(self.trans_mat)
        modulus = np.abs(eigvals)
        print(modulus)

        total = np.sum(modulus > 1.)

        if total > 0:
            raise ValueError("not stabilizable")


    def reset(self, state):
        # resets the starting of the LQR system to the given state
        # state: numpy array of dimension d representing the state
        self.state = state


    def step(self):
        # takes a step according to the policy
        # returns
        # initial_state: numpy vector of size state_dim, representing starting state
        # cost: float representing the cost of this action
        # next_state: the state transitioned to

        cost = self.state.T @ self.cost_mat @ self.state
        initial_state = self.state
        self.state = self.trans_mat @ self.state \
                        + np.random.multivariate_normal(np.zeros(self.state_dim), self.noise_cov)
        next_state = self.state

        return initial_state, cost, next_state


    # EX: this doesn't work for some reason, debug later and use value iteration for now
    def compute_value(self):
        # computes the value function associated with the LQR

        # reshapes things appropriately
        tensor_trans = np.tensordot(self.trans_mat.T, self.trans_mat.T)
        solve_mat = np.eye(self.state_dim * self.state_dim, self.state_dim * self.state_dim) \
                        - gamma * tensor_trans
        cost_vec = self.cost_mat.flatten()

        # computes the solution
        P_vec = npla.solve(solve_mat, cost_vec)
        self.value_P = np.reshape(P_vec, (self.state_dim, self.state_dim))
        self.value_const = np.trace(self.value_P @ self.noise_cov) * gamma / (1 - gamma)
        return self.value_P, self.value_const


    def value_iteration(self, num_iters = 100):
        # runs value iteration to compute the value function
        # num_iters: integer representing number of updates
        # 
        # returns
        # value_P: numpy matrix of dimension d x d representing the matrix in value function
        # value_const: float representing the constant in the value function

        P = np.zeros((self.state_dim, self.state_dim))
        for i in range(num_iters):
            P = self.gamma * self.trans_mat.T @ P @ self.trans_mat + self.cost_mat

        self.value_P = P
        self.value_const = np.trace(self.value_P @ self.noise_cov) * gamma / (1 - gamma)

        return self.value_P, self.value_const


    def compute_value(self, state):
        # computes the value of the state, assumes that self.value_P and self.value_const
        # have been computed already
        # state: numpy array of size state_dim representing the state to evaluate
        #
        # returns
        # value: float denoting the value of the state

        return state.T @ self.value_P @ state + self.value_const


    def compute_Bellman(self, state, value_func, num_samples = 10000):
        # computes the Bellman update of a given state using value_func
        # state: numpy array of size state_dim representing the state to compute Bellman update of
        # value_func: objet of the LSTD_Boost class
        #              a value function to compute the Bellman update of
        # num_samples: int representing then number of samples to perform Monte Carlo with
        # 
        # returns
        # value: float, Bellman update of value_func at the current state

        cost = state.T @ self.cost_mat @ state
        next_states = np.random.multivariate_normal(np.zeros(self.state_dim), 
                            self.noise_cov, size=num_samples) + self.trans_mat @ state
        next_values = value_func.evaluate(next_states)
        return cost + self.gamma * np.average(next_values)




##########################################################


def collect_samples(LQR_instance, num_samples = 1000):
    # collects samples into an array format
    # LQR_instance: object of the LQR class to get samples from
    # num_samples: integer representing number of samples desired
    #
    # returns
    # curr_state: numpy matrix of dimension num_samples x state_dim representing
    #               the starting state in a transition
    # costs: numy array of size num_samples representing costs in a transition
    # next_state: numpy matrix of dimension num_samples x state_dim representing
    #                finishing state in a transition

    curr_states = np.zeros((num_samples, LQR_instance.state_dim))
    next_states = np.zeros((num_samples, LQR_instance.state_dim))
    costs = np.zeros(num_samples)

    for i in range(num_samples):
        state, cost, next_state = LQR_instance.step()

        curr_states[i, :] = state
        costs[i] = cost
        next_states[i, :] = next_state

    return curr_states, costs, next_states


##########################################################


def evaluate_value_boost(LQR_instance, value_function, num_samples = 10000):
    # evalutes how good the value function is
    # LQR_instance: object of the LQR class to get samples from, 
    #               assumes that value_P has already been computed
    # value_function: object of the LSTD_Boost classes, assumes already run
    # num_samples: number of samples used to evaluate the value function

    states, _, _ = collect_samples(LQR_instance, num_samples)
    true_values = np.zeros(num_samples)

    pred_values = value_function.evaluate(states)

    for i in range(num_samples):
        true_values[i] = LQR_instance.compute_value(states[i, :])

    return mean_squared_error(true_values, pred_values)


def evaluate_value_func(LQR_instance, value_mat, value_cons, num_samples = 25000):
    # evalutes how good the value function is given a matrix and constant representing
    #           the value function
    # LQR_instance: object of the LQR class to get samples from, 
    #               assumes that value_P has already been computed
    # value_mat: numpy matrix of dimension state_dim x state_dim, representing P
    # value_cons: float representing constant cost, c
    # num_samples: number of samples used to evaluate the value function

    states, _, _ = collect_samples(LQR_instance, num_samples)
    true_values = np.zeros(num_samples)
    pred_values = np.zeros(num_samples)

    for i in range(num_samples):
        state = states[i, :]
        pred_values[i] = state @ value_mat @ state + value_cons

        true_values[i] = LQR_instance.compute_value(states[i, :])

    return mean_squared_error(true_values, pred_values)


def evaluate_oos(LQR_instance, basis, num_check = 1000):

    #check if the regression step is working
    bellman_values = np.zeros(num_check)
    obs, cost, next_obs = collect_samples(LQR_example, num_check)

    for i in range(num_check):
        bellman_values[i] = LQR_example.compute_Bellman(obs[i, :], value_func)

    bellman_values = bellman_values - value_func.evaluate(obs)

    pred_values = basis.predict(obs)

    print("out of sample MSE is")
    print(mean_squared_error(bellman_values, pred_values))


##########################################################

# Debugging examples



state_dim = 5
action_dim = 3
gamma = 0.9

np.random.seed(2022)

A = np.random.rand(state_dim, state_dim) / 20
B = np.random.rand(state_dim, action_dim) / 20
K = np.random.rand(action_dim, state_dim)
noise_sd = np.eye(state_dim)
Q = np.random.rand(state_dim, state_dim)
Q = Q.T @ Q
R = np.random.rand(action_dim, action_dim)
R = R.T @ R


LQR_example = LQR(A, B, K, noise_sd, Q, R, gamma)
LQR_example.reset(np.random.rand(state_dim))

mat, const = LQR_example.value_iteration(50000)
init_error = evaluate_value_func(LQR_example, np.zeros((state_dim, state_dim)), 0)

num_iter = 30

##########################################################


# runs value iteration
value_iter_error = np.zeros(num_iter)
value_mat = np.zeros((state_dim, state_dim))
value_cons = 0.

value_iter_error[0] = init_error

for i in range(1, num_iter):
    value_cons = gamma * value_cons + gamma * np.trace(value_mat @ LQR_example.noise_cov)
    value_mat = LQR_example.cost_mat + gamma * LQR_example.trans_mat.T @ value_mat @ LQR_example.trans_mat

    value_iter_error[i] = evaluate_value_func(LQR_example, value_mat, value_cons)


##########################################################

value_func = XG_LSTD_Boost(LQR_example.state_dim, discount = gamma, num_regressors = 300, max_depth = 7)
obs, cost, next_obs = collect_samples(LQR_example, 30000)
value_func.construct_first(obs, cost)
value_func.construct_LSTD(obs, cost, next_obs)



# trains the Krylov-Bellman boosting algorithm
debias = False


lstd_boost_error = np.zeros(num_iter)
lstd_boost_error[0] = init_error
lstd_boost_error[1] = evaluate_value_boost(LQR_example, value_func)

for i in range(2, num_iter):
    print("on iteration", i)

    obs, cost, next_obs = collect_samples(LQR_example, int(35000 * np.sqrt(i)))
    value_func.construct_next(obs, cost, next_obs, 
                                num_regressors = int(4 + i // 5), 
                                max_depth = 2, debias = debias)

    # obs, cost, next_obs = collect_samples(LQR_example, int(50000 * (1 + np.sqrt(i) / 5)))
    value_func.construct_LSTD(obs, cost, next_obs)

    error = evaluate_value_boost(LQR_example, value_func)
    lstd_boost_error[i] = error
    print(error)



##########################################################

# runs fitted value iteration

fvi_func = XG_fitted_value(LQR_example.state_dim, gamma)
obs, cost, next_obs = collect_samples(LQR_example, 35000)
fvi_func.first_fvi(obs, cost, num_regressors = 4, max_depth = 2)

fvi_error = np.zeros(num_iter)
fvi_error[0] = init_error
fvi_error[1] = evaluate_value_boost(LQR_example, fvi_func)

for i in range(2, num_iter):
    print("on iteration", i)

    obs, cost, next_obs = collect_samples(LQR_example, int(35000 * np.sqrt(i)))
    fvi_func.fitted_value(obs, cost, next_obs, num_regressors = int(4 + i // 5), 
                                max_depth = 2)

    fvi_error[i] = evaluate_value_boost(LQR_example, fvi_func)

##########################################################

# plt.plot(np.log(value_iter_error), "ro--", label = "Value Iteration")
# plt.plot(np.log(lstd_boost_error), "bo--", label = "Krylov-Bellman Boosting")
# plt.plot(np.log(fvi_error), "go--", label = "Fitted Value Iteration")
# plt.legend()
# plt.xlabel("Number of iterations", fontsize = 18)
# plt.ylabel("Log error", fontsize = 18)
# plt.title("LQR policy evaluation, $\gamma = 0.99$", fontsize = 23)
# plt.savefig("plots/plot.pdf", dpi=300)
# plt.show()

##########################################################


# saves data
path = "../data/"
filename = "LQR" + str(gamma) + "_numiter" + str(num_iter) + "_"

np.savetxt(path + filename + "valueiter.csv", value_iter_error, delimiter = ",")
np.savetxt(path + filename + "fvi.csv", fvi_error, delimiter = ",")
np.savetxt(path + filename + "lstdboost.csv", lstd_boost_error, delimiter = ",")




