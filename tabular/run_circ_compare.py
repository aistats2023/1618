import numpy as np
import numpy.linalg as npl
import numpy.random as npr

from scipy.linalg import eig 
import scipy.linalg as scpl

import matplotlib.pyplot as plt

from lstd_boost import *
from fitted_value import *

from mdp import tabular


###############################################################################################


gamma = 0.9
dim = 200

# Generate circulant matrix of random walk
rowvec = np.zeros(dim)

# move length in random walk
L = 3
rowvec[0:L] = 1 / L

Pmat = scpl.circulant(rowvec)
Pmat = 0.5 * (Pmat + Pmat.T)

reward = npr.rand(dim)

mrp = tabular(Pmat, reward, gamma)

tol = 0.1
init_error = mrp.evaluate_value(np.zeros(dim))


##########################################################

# runs value iteration

value_iter_error = [init_error]
value_iter = np.zeros(dim)
i = 1

while True:
    value_iter = reward + gamma * Pmat @ value_iter
    value_iter_error.append(mrp.evaluate_value(value_iter))

    if value_iter_error[i] / value_iter_error[0] < tol:
        break

    i += 1


##########################################################

# runs fitted value iteration

fvi_error = [init_error]
fvi_func = tabular_fitted_value(dim, gamma)
i = 1


while True:
    obs, rew, next_obs = mrp.collect_samples(int(3000 * np.sqrt(i)))    

    fvi_func.fitted_value(obs, rew, next_obs)
    fvi_error.append(mrp.evaluate_value(fvi_func.compute_value()))

    if fvi_error[i] / fvi_error[0] < tol:
        break

    i += 1

##########################################################

# runs Krylov Bellman Boosting

value_func = tabular_LSTD_Boost(dim, gamma)
obs, rew, next_obs = mrp.collect_samples(3000)
value_func.construct_first(obs, rew)
obs, rew, next_obs = mrp.collect_samples(1000)
value_func.construct_LSTD(obs, rew, next_obs)

lstd_boost_error = [init_error, 0]
lstd_boost_error[1] = mrp.evaluate_value(value_func.compute_value())

i = 2

run = True

if lstd_boost_error[1] / lstd_boost_error[0] < tol:
        run = False

while run:
    obs, rew, next_obs = mrp.collect_samples(int(3000 * np.sqrt(i)))
    value_func.construct_next(obs, rew, next_obs)

    # obs, rew, next_obs = mrp.collect_samples(10000)
    value_func.construct_LSTD(obs, rew, next_obs)

    lstd_boost_error.append(mrp.evaluate_value(value_func.compute_value()))

    if lstd_boost_error[i] / lstd_boost_error[0] < tol:
        break

    i += 1



##########################################################

# saves the data
path = "../data/"
filename = "circ" + str(gamma) + "_tol" + str(tol) + "_"

np.savetxt(path + filename + "valueiter.csv", value_iter_error, delimiter = ",")
np.savetxt(path + filename + "fvi.csv", fvi_error, delimiter = ",")
np.savetxt(path + filename + "lstdboost.csv", lstd_boost_error, delimiter = ",")