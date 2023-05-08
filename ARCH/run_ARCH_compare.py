# file runs policy evaluation for infinite horizon ARCH with discounting
# implements an ARCH system that is comptabile with lstd_boost

import numpy as np
import pandas as pd
import numpy.linalg as npla
import numpy.random as npra
import scipy.linalg as scila
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from lstd_boost import *
from fitted_value import *

##########################################################

class ARCH():
    # object that represents an ARCH instance with quadratic costs

    def __init__(self, A, q0, Q, noise_cov, R, gamma):
        # initializes the ARCH system
        # A: numpy float matrix (dimension d x d) representing system dynamics
        # q0: float representing constant in autoregressive noise
        # Q: numpy float PSD matrix (dimension d x d) representing state effect in noise
        # noise_cov: numpy float PSD matrix (dimension d x d) representing covariance of noise
        # R: numpy float PSD matrix (dimension d x d) representing state cost

        # sets parameters
        self.A = A
        self.q0 = q0
        self.Q = Q
        self.noise_cov = noise_cov
        self.R = R
        self.gamma = gamma

        self.state_dim = np.shape(A)[0]


    def reset(self, state):
        # resets the starting of the ARCH system to the given state
        # state: numpy array of dimension d representing the state
        self.state = state


    def step(self):
        # takes a step in the system
        # 
        # returns
        # initial_state: numpy float vector (diminesion state_dim), representing starting state
        # cost: float representing cost of this state
        # next_state: the state transitioned to

        cost = self.state.T @ self.R @ self.state
        initial_state = self.state
        self.state = self.A @ self.state + np.sqrt(self.q0 + self.state.T @ self.Q @ self.state.T) \
                                                * npra.multivariate_normal(np.zeros(self.state_dim), self.noise_cov)
        next_state = self.state

        return initial_state, cost, next_state


    def value_iteration(self, num_iters = 1000):
        # runs value iteration to compute the value function
        # num_iters: integer representing number of iterations
        #
        # returns
        # value_P: numpy float matrix (dimension d x d) represnting quadratic part of value function
        # value_const: float for constant in value function

        P = np.zeros((self.state_dim, self.state_dim))
        for i in range(num_iters):
            P = self.gamma * self.A.T @ P @ self.A + self.gamma  * np.trace(P @ self.noise_cov) * self.Q + self.R

        self.value_P = P
        self.value_const = self.q0 * self.gamma * np.trace(self.value_P @ self.noise_cov) / (1 - self.gamma)

        return self.value_P, 0.


    def compute_value(self, state):
        # computes the value of the state, assumes that self.value_P and 
        # self.value_const have been computed already
        # state: numpy float array (dimension d), state to evaluate
        #
        # returns
        # value: float, value of the state

        try:
            value = state.T @ self.value_P @ state + self.value_const
            return value
        except NameError:
            print("value can't be computed yet")
            return 0.


##########################################################

def collect_samples(ARCH_instance, num_samples = 1000):
    # collects samples into an array format
    # ARCH_instance: object of the ARCH class to get samples from
    # num_samples: integer representing number of samples desired
    #
    # returns
    # curr_state: numpy matrix of dimension num_samples x state_dim representing
    #               the starting state in a transition
    # costs: numy array of size num_samples representing costs in a transition
    # next_state: numpy matrix of dimension num_samples x state_dim representing
    #                finishing state in a transition

    curr_states = np.zeros((num_samples, ARCH_instance.state_dim))
    next_states = np.zeros((num_samples, ARCH_instance.state_dim))
    costs = np.zeros(num_samples)

    for i in range(num_samples):
        state, cost, next_state = ARCH_instance.step()

        curr_states[i, :] = state
        costs[i] = cost
        next_states[i, :] = next_state

    return curr_states, costs, next_states


##########################################################


def evaluate_value_boost(ARCH_instance, value_function, num_samples = 10000):
    # evalutes how good the value function is
    # ARCH_instance: object of the ARCH class to get samples from, 
    #               assumes that value_P has already been computed
    # value_function: object of the LSTD_Boost classes, assumes already run
    # num_samples: number of samples used to evaluate the value function

    states, _, _ = collect_samples(ARCH_instance, num_samples)
    true_values = np.zeros(num_samples)

    pred_values = value_function.evaluate(states)

    for i in range(num_samples):
        true_values[i] = ARCH_instance.compute_value(states[i, :])

    return mean_squared_error(true_values, pred_values)


def evaluate_value_func(ARCH_instance, value_mat, value_cons, num_samples = 25000):
    # evalutes how good the value function is given a matrix and constant representing
    #           the value function
    # ARCH_instance: object of the ARCH class to get samples from, 
    #               assumes that value_P has already been computed
    # value_mat: numpy matrix of dimension state_dim x state_dim, representing P
    # value_cons: float representing constant cost, c
    # num_samples: number of samples used to evaluate the value function

    states, _, _ = collect_samples(ARCH_instance, num_samples)
    true_values = np.zeros(num_samples)
    pred_values = np.zeros(num_samples)

    for i in range(num_samples):
        state = states[i, :]
        pred_values[i] = state @ value_mat @ state + value_cons

        true_values[i] = ARCH_instance.compute_value(states[i, :])

    return mean_squared_error(true_values, pred_values)


##########################################################

# Debugging examples


state_dim = 5
gamma = 0.9

np.random.seed(2022)

A = np.random.rand(state_dim, state_dim) / 10
q0 = 0.5
Q = np.random.rand(state_dim, state_dim) / 10
Q = Q.T @ Q
R = np.random.rand(state_dim, state_dim)
noise_sd = np.eye(state_dim)

ARCH_example = ARCH(A, q0, Q, noise_sd, R, gamma)
ARCH_example.reset(np.random.rand(state_dim))

mat, const = ARCH_example.value_iteration(50000)

init_error = evaluate_value_func(ARCH_example, np.zeros((state_dim, state_dim)), 0)
tol = 0.1


##########################################################


# runs value iteration
value_iter_error = [init_error]
value_mat = np.zeros((state_dim, state_dim))
value_cons = 0.

i = 1

while True:
    value_cons = gamma * value_cons + gamma * q0 * np.trace(value_mat @ noise_sd)
    value_mat = R + gamma * A.T @ value_mat @ A + gamma * np.trace(Q @ noise_sd)

    value_iter_error.append(evaluate_value_func(ARCH_example, value_mat, value_cons))

    if value_iter_error[i] / init_error < tol:
        break

    i += 1



##########################################################

value_func = XG_LSTD_Boost(ARCH_example.state_dim, discount = gamma, num_regressors = 300, max_depth = 7)
obs, cost, next_obs = collect_samples(ARCH_example, 20000)
value_func.construct_first(obs, cost)
value_func.construct_LSTD(obs, cost, next_obs)



# trains the Krylov-Bellman boosting algorithm
debias = False

lstd_boost_error = [init_error, 0]
lstd_boost_error[1] = evaluate_value_boost(ARCH_example, value_func)

i = 2

while True:
    print("on iteration", i)

    obs, cost, next_obs = collect_samples(ARCH_example, int(20000 * np.sqrt(i)))
    value_func.construct_next(obs, cost, next_obs, 
                                num_regressors = int(4 + i // 5), 
                                max_depth = 2, debias = debias)

    # obs, cost, next_obs = collect_samples(ARCH_example, int(50000 * (1 + np.sqrt(i) / 5)))
    value_func.construct_LSTD(obs, cost, next_obs)

    error = evaluate_value_boost(ARCH_example, value_func)
    lstd_boost_error.append(error)
    print(error)

    if error / init_error < tol:
        break

    i += 1


##########################################################

# runs fitted value iteration
print("Now on fitted value iteration")

fvi_func = XG_fitted_value(ARCH_example.state_dim, gamma)
obs, cost, next_obs = collect_samples(ARCH_example, 20000)
fvi_func.first_fvi(obs, cost, num_regressors = 4, max_depth = 2)

fvi_error = [init_error, 0]
fvi_error[1] = evaluate_value_boost(ARCH_example, fvi_func)

i = 2

while True:
    print("on iteration", i)

    obs, cost, next_obs = collect_samples(ARCH_example, int(20000 * np.sqrt(i)))
    fvi_func.fitted_value(obs, cost, next_obs, num_regressors = int(4 + i // 5), 
                                max_depth = 2)

    fvi_error.append(evaluate_value_boost(ARCH_example, fvi_func))

    if fvi_error[i] / init_error < tol:
        break

    i += 1


##########################################################

# saves the data
path = "../data/"
filename = "ARCH" + str(gamma) + "_tol" + str(tol) + "_"

np.savetxt(path + filename + "valueiter.csv", value_iter_error, delimiter = ",")
np.savetxt(path + filename + "fvi.csv", fvi_error, delimiter = ",")
np.savetxt(path + filename + "lstdboost.csv", lstd_boost_error, delimiter = ",")