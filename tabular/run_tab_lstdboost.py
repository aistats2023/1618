import numpy as np
import numpy.linalg as npl
import numpy.random as npr

from scipy.linalg import eig 

import matplotlib.pyplot as plt

from lstd_boost import *
from fitted_value import *

from mdp import tabular


###############################################################################################

# generate random MRP
gamma = 0.99
dim = 300

Pmat = npr.rand(dim, dim)
# print(np.sum(Pmat, axis = 1))
reward = npr.rand(dim)

mrp = tabular(Pmat, reward, gamma)
init_error = mrp.evaluate_value(np.zeros(dim))

##########################################################

# runs value iteration

num_iter = 30

value_iter_error = np.zeros(num_iter)
value_iter = np.zeros(dim)

value_iter_error[0] = init_error

for i in range(1, num_iter):
    value_iter = reward + gamma * Pmat @ value_iter
    value_iter_error[i] = mrp.evaluate_value(value_iter)

##########################################################

# runs fitted value iteration

fvi_error = np.zeros(num_iter)
fvi_error[0] = init_error

fvi_func = tabular_fitted_value(dim, gamma)

for i in range(1, num_iter):
    obs, rew, next_obs = mrp.collect_samples(int(3000 * np.sqrt(i)))

    fvi_func.fitted_value(obs, rew, next_obs)
    fvi_error[i] = mrp.evaluate_value(fvi_func.compute_value())

##########################################################

# runs Krylov Bellman Boosting

value_func = tabular_LSTD_Boost(dim, gamma)
obs, rew, next_obs = mrp.collect_samples(3000)
value_func.construct_first(obs, rew)
value_func.construct_LSTD(obs, rew, next_obs)

lstd_boost_error = np.zeros(num_iter)
lstd_boost_error[0] = init_error
lstd_boost_error[1] = mrp.evaluate_value(value_func.compute_value())

for i in range(2, num_iter):

    obs, rew, next_obs = mrp.collect_samples(int(3000 * np.sqrt(i)))
    value_func.construct_next(obs, rew, next_obs)

    # obs, rew, next_obs = mrp.collect_samples(10000)
    value_func.construct_LSTD(obs, rew, next_obs)

    lstd_boost_error[i] = mrp.evaluate_value(value_func.compute_value())

##########################################################

# saves data
path = "../data/"
filename = "tabular" + str(gamma) + "_numiter" + str(num_iter) + "_"

np.savetxt(path + filename + "valueiter.csv", value_iter_error, delimiter = ",")
np.savetxt(path + filename + "fvi.csv", fvi_error, delimiter = ",")
np.savetxt(path + filename + "lstdboost.csv", lstd_boost_error, delimiter = ",")
