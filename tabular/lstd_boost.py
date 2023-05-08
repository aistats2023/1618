import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
import numpy as np
import pandas as pd
import gym

from xgboost import *

from sklearn.metrics import mean_squared_error

from stable_baselines3 import A2C


##########################################################

class LSTD_Boost():
	# object that represents the algorithm
	# honestly maybe using neural networks is not the best

	def __init__(self, state_dim, discount = 0.99, hidden_dim = 256, learning_rate = 0.001):
		# initializes the object for running the algorithm
		# state_dim: int representing the dimension of the state space

		self.state_size = state_dim
		self.discount = discount
		self.hidden_dim = hidden_dim
		self.learning_rate = learning_rate

		# list of neural networks that represent that constitue linear function representation
		self.lin_rep = [lambda x: 0.]
		self.weights = np.array([1])
		self.num_basis = 1


	def construct_first(self, observations, rewards, batch_size = 64, iters = 10000):
		# constructs the initial basis, an estimate of the reward function

		basis = ValueNetwork(self.state_size, self.hidden_dim, 1)
		optimizer = optim.RMSprop(basis.parameters(), lr = self.learning_rate)
		num_samples = observations.shape[0]

		# observations = torch.tensor(observations, dtype=torch.float)
		# output = torch.tensor(rewards[:, None], dtype=torch.float)

		for _ in range(iters):
			optimizer.zero_grad()

			batch_ind = np.random.randint(num_samples, size = batch_size)
			batch_obs = torch.tensor(observations[batch_ind, :], dtype = torch.float)
			batch_out = torch.tensor(rewards[batch_ind, None], dtype=torch.float)

			values = basis(batch_obs)
			loss_value = 1 * F.mse_loss(values, batch_out)
			loss_value.backward()
			optimizer.step()


		self.lin_rep[0] = basis


	def construct_LSTD(self, observations, rewards, next_observations):
		# updates self.weights to match the LSTD fit

		curr_feature_mat = self.compute_feature_matrix(observations)
		next_feature_mat = self.compute_feature_matrix(next_observations)

		cross_covmat = curr_feature_mat.T @ (curr_feature_mat - self.discount * next_feature_mat)
		reward_feat = curr_feature_mat.T @ rewards

		self.weights = np.linalg.solve(cross_covmat, reward_feat)

		return self.weights

	def construct_next(self, observations, rewards, next_observations, batch_size = 64, iters = 10000):
		curr_values = self.evaluate(observations)
		num_samples = observations.shape[0]

		rewards = torch.tensor(rewards, dtype=torch.float)

		next_values = torch.add(torch.reshape(rewards, (num_samples, 1)), 
								self.evaluate(next_observations), alpha = self.discount)
		outputs = torch.sub(next_values, curr_values)

		basis = ValueNetwork(self.state_size, self.hidden_dim, 1)
		optimizer = optim.RMSprop(basis.parameters(), lr = self.learning_rate)
		
		for _ in range(iters):
			optimizer.zero_grad()

			batch_ind = np.random.randint(num_samples, size = batch_size)
			batch_obs = torch.tensor(observations[batch_ind, :], dtype=torch.float)
			batch_out = torch.tensor(outputs[batch_ind, :], dtype=torch.float)

			values = basis(batch_obs)
			loss_value = 1 * F.mse_loss(values, batch_out)
			loss_value.backward()
			optimizer.step()

		self.lin_rep.append(basis)
		self.weights = np.append(self.weights, 1)
		self.num_basis += 1


	def evaluate(self, observations):
		# returns the current value function

		values = self.lin_rep[0](torch.tensor(observations, dtype = torch.float)) * self.weights[0]

		for i in range(1, self.num_basis):
			values += self.lin_rep[i](torch.tensor(observations, dtype=torch.float)) * self.weights[i]

		return values


	def compute_feature_matrix(self, observations):
		# computes the matrix of feature representations
		num_samples = observations.shape[0]

		feature_mat = np.empty((num_samples, self.num_basis))

		for i in range(self.num_basis):
			feature = self.lin_rep[i](torch.tensor(observations, dtype=torch.float))
			feature_mat[:, i] = feature.detach().cpu().numpy().flatten()

		return feature_mat




##########################################################

class ValueNetwork(nn.Module):
    # defines the neural network for policy evaluation

    def __init__(self, input_size, hidden_size, output_size):
        # defines 3(?) layer neural network
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


##########################################################



class XG_LSTD_Boost():
	# object that represents the algorithm
	# honestly maybe using neural networks is not the best

	def __init__(self, state_dim, discount = 0.99, num_regressors = 300, max_depth=7):
		# initializes the object for running the algorithm
		# state_dim: int representing the dimension of the state space

		self.state_size = state_dim
		self.discount = discount
		self.num_regressors = num_regressors
		self.max_depth = max_depth

		# list of neural networks that represent that constitue linear function representation
		self.lin_rep = [lambda x: 0.]
		self.weights = np.array([1])
		self.num_basis = 1


	def construct_first(self, observations, rewards, num_regressors = 300, max_depth = 7):
		# constructs the initial basis, an estimate of the reward function


		basis = XGBRegressor(n_estimators = num_regressors, max_depth = max_depth)
		basis.fit(observations, rewards)

		self.lin_rep[0] = basis


	def construct_LSTD(self, observations, rewards, next_observations):
		# updates self.weights to match the LSTD fit

		curr_feature_mat = self.compute_feature_matrix(observations)
		next_feature_mat = self.compute_feature_matrix(next_observations)

		cross_covmat = curr_feature_mat.T @ (curr_feature_mat - self.discount * next_feature_mat)
		reward_feat = curr_feature_mat.T @ rewards

		self.weights = np.linalg.solve(cross_covmat, reward_feat)

		return self.weights


	def construct_next(self, observations, rewards, next_observations, 
						num_regressors = 300, max_depth = 7, iters = 10000, 
						debias = False):
		curr_values = self.evaluate(observations)
		num_samples = observations.shape[0]


		next_values = rewards + self.discount * self.evaluate(next_observations) - curr_values

		basis = XGBRegressor(n_estimators = num_regressors, 
								max_depth = max_depth, subsample = 1)
		basis.fit(observations, next_values)

		if debias:
			pass

		self.lin_rep.append(basis)
		self.weights = np.append(self.weights, 1)
		self.num_basis += 1


	def evaluate(self, observations):
		# returns the current value function

		values = self.lin_rep[0].predict(observations) * self.weights[0]

		for i in range(1, self.num_basis):
			values += self.lin_rep[i].predict(observations) * self.weights[i]

		return values


	def compute_feature_matrix(self, observations):
		# computes the matrix of feature representations
		num_samples = observations.shape[0]

		feature_mat = np.empty((num_samples, self.num_basis))

		for i in range(self.num_basis):
			feature = self.lin_rep[i].predict(observations)
			feature_mat[:, i] = feature

		return feature_mat


##########################################################

class tabular_LSTD_Boost():

	# LSTD Boost for tabular MDPs

	def __init__(self, num_states, discount = 0.99):
		# initializes the object
		# num_states: int representing the number of states in the MRP
		# discount: float, discount factor for MRP

		self.num_states = num_states
		self.discount = discount


		# list of neural networks that represent that constitue linear function representation
		self.lin_rep = [lambda x: 0.]
		self.weights = np.array([1])
		self.num_basis = 1


	def construct_first(self, observations, rewards):
		# constructs the initial basis function, an estimate of the reward function
		# observations: numpy int array of length N representing the states observed,
		#				assumes the states are indexed 0 through num_states
		# rewards: numpy float array of length N representing the rewards observed


		basis = np.zeros(self.num_states)
		counts = np.zeros(self.num_states) + 1

		num_samples = np.shape(observations)[0]

		# computes average reward 
		for i in range(num_samples):
			ind = int(observations[i])
			basis[ind] = rewards[i]
			counts[ind] += 1

		basis = basis / counts

		self.weights = np.array([1])
		self.num_basis = 1
		self.lin_rep = basis.reshape((self.num_states, 1))


	def construct_next(self, observations, rewards, next_observations):
		# updates self.weights to match the LSTD fit
		# observations: numpy int array of length N representing the states observed,
		#				assumes the states are indexed 0 through num_states
		# rewards: numpy float array of length N representing the rewards observed
		# next_observations: numpy int array of length N representing the next states 
		#				observed in the tuple, assumes the states are indexed 0 through num_states

		curr_values = self.evaluate(observations)
		num_samples = observations.shape[0]

		next_values = rewards + self.discount * self.evaluate(next_observations) - curr_values

		basis = np.zeros(self.num_states)
		counts = np.zeros(self.num_states) + 1

		num_samples = np.shape(observations)[0]

		# computes average reward 
		for i in range(num_samples):
			ind = int(observations[i])
			basis[ind] += next_values[i]
			counts[ind] += 1

		basis = basis / counts
		basis = basis.reshape((self.num_states, 1))

		self.lin_rep = np.hstack((self.lin_rep, basis))
		self.weights = np.append(self.weights, 1)
		self.num_basis += 1


	def construct_LSTD(self, observations, rewards, next_observations):
		# updates self.weights to match the LSTD fit
		# observations: numpy int array of length N representing the states observed,
		#				assumes the states are indexed 0 through num_states
		# rewards: numpy float array of length N representing the rewards observed
		# next_observations: numpy int array of length N representing the next states 
		#				observed in the tuple, assumes the states are indexed 0 through num_states

		curr_feature_mat = self.compute_feature_matrix(observations)
		next_feature_mat = self.compute_feature_matrix(next_observations)

		cross_covmat = curr_feature_mat.T @ (curr_feature_mat - self.discount * next_feature_mat)
		reward_feat = curr_feature_mat.T @ rewards

		self.weights = np.linalg.solve(cross_covmat, reward_feat)

		return self.weights


	def compute_feature_matrix(self, observations):
		# computes the matrix of feature representations
		# observations: numpy int array of length N representing the states observed,
		#				assumes the states are indexed 0 through num_states
		#
		# returns
		# feature_mat: numpy float matrix of dim (num_samples, num_basis) representing
		# 				the feature matrix

		num_samples = observations.shape[0]

		feature_mat = np.zeros((num_samples, self.num_basis))

		for i in range(num_samples):
			ind = int(observations[i])
			feature_mat[i, :] = self.lin_rep[ind, :]


		return feature_mat



	def evaluate(self, observations):
		# computes the value function of the observations
		# observations: numpy int array of length N representing the states observed,
		#				assumes the states are indexed 0 through num_states
		#
		# returns
		# value_mat: numpy float array containing the value functions of the observations

		values = self.lin_rep @ self.weights
		num_samples = np.shape(observations)[0]

		value_mat = np.zeros(num_samples)

		for i in range(num_samples):
			ind = int(observations[i])
			value_mat[i] = values[ind]

		return value_mat


	def compute_value(self):
		# computes the vector representing the value function
		#
		# returns
		# value: numpy float array representing the value function

		return self.evaluate(np.arange(self.num_states))



##########################################################

def generate_samples(env, model, batch_size):
    # generates the samples needed for each batch

    actions = np.empty((batch_size,), dtype=int)
    dones = np.empty((batch_size,), dtype=bool)
    rewards = np.empty((batch_size,), dtype=float)
    observations = np.empty((batch_size,) + env.observation_space.shape, dtype = float)
    next_observations = np.empty((batch_size,) + env.observation_space.shape, dtype = float)

    observation = env.reset()

    for i in range(batch_size):
        observations[i] = observation
        policy, _ = model.predict(observation, deterministic=True)
        observation, rewards[i], dones[i], _ = env.step(policy)
        next_observations[i] = observation

        if dones[i]:
            observation = env.reset()


    return observations, next_observations, rewards, dones

##########################################################



