import numpy as np

from xgboost import *

from sklearn.metrics import mean_squared_error

##########################################################

class tabular_fitted_value():

    # fitted value for tabular MDPs

    def __init__(self, num_states, discount = 0.99):
        # initializes the object
        # num_states: int representing the number of states in the MRP
        # discount: float, discount factor for MRP

        self.num_states = num_states
        self.discount = discount

        self.value = np.zeros(self.num_states)


    def fitted_value(self, observations, rewards, next_observations):
        # runs a fitted value iteration update with given samples
        # observations: numpy int array of length N representing the states observed,
        #               assumes the states are indexed 0 through num_states
        # rewards: numpy float array of length N representing the rewards observed
        # next_observations: numpy int array of length N representing the next states 
        #               observed in the tuple, assumes the states are indexed 0 through num_states

        next_values = rewards + self.discount * self.evaluate(next_observations) 

        value_update = np.zeros(self.num_states)
        counts = np.zeros(self.num_states) + 1

        num_samples = observations.shape[0]

        for i in range(num_samples):
            ind = int(observations[i])
            value_update[ind] += next_values[i] 
            counts[ind] += 1

        value_update = value_update / counts

        self.value = value_update


    def evaluate(self, observations):
        # computes the value function of the observations
        # observations: numpy int array of length N representing the states observed,
        #               assumes the states are indexed 0 through num_states
        #
        # returns
        # value_mat: numpy float array containing the value functions of the observations

        num_samples = np.shape(observations)[0]

        value_mat = np.zeros(num_samples)

        for i in range(num_samples):
            ind = int(observations[i])
            value_mat[i] = self.value[ind]

        return value_mat


    def compute_value(self):
        # computes the vector representing the value function
        #
        # returns
        # value: numpy float array representing the value function

        return self.value     

##########################################################


class XG_fitted_value():

    # fitted value using XGBoost as the regression step

    def __init__(self, state_dim, discount = 0.99):
        # initializes the object for running
        # state_dim: int for dimension of state space
        # discount: float for discount factor
        
        self.state_size = state_dim
        self.discount = discount


    def first_fvi(self, observations, rewards, num_regressors = 300, max_depth = 7):
        # runs first update of the fitted-value iteration algorithm
        # observations: numpy float matrix of size num_samples x state_dim, representing
        #                   the current state in observation tuples
        # rewards: numpy float array of size num_samples, reward observed in the transition

        self.value = XGBRegressor(n_estimators = num_regressors, max_depth = max_depth)
        self.value.fit(observations, rewards)


    def fitted_value(self, observations, rewards, next_observations, 
                        num_regressors = 300, max_depth = 7):
        # runs one update of the fitted-value iteration algorithm
        # observations: numpy float matrix of size num_samples x state_dim, representing
        #                   the current state in observation tuples
        # rewards: numpy float array of size num_samples, reward observed in the transition
        # next_observations: numpy float matrix of size num_samples x state_dim, representing
        #                   the next state in observation tuples


        next_values = rewards + self.discount * self.evaluate(next_observations)

        self.value = XGBRegressor(n_estimators = num_regressors, max_depth = max_depth)
        self.value.fit(observations, next_values)


    def evaluate(self, observations):
        # evaluates the value function at the given observations
        # observations: numpy float matrix of size num_samples x state_dim, representing
        #                   the current state in observation tuples
        # returns: numpy float array of size num_samples, value function evaluated at obs

        return self.value.predict(observations)