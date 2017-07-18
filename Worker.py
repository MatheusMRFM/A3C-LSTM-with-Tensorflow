import numpy as np
import tensorflow as tf
import time, random, threading
import multiprocessing
import scipy.misc
from Network import *
from Env_Doom import *
from Env_Atari import *

SAVER_INTERVAL = 100
RND_ACTION_PCT = 0.0

ATARI	= 0
DOOM 	= 1


"""
Calulates the discounted rewards for each timestep t in the batch, where:
	- Rt = \sum_{i=0}^{k} gamma^{i} * r_{t+i}
This results in a 1-step discounted reward for timestep B, 2-step discounted
reward for timestep B-1,...., B-step discounted reward for timestep 1, where:
	- B = Batch Size
"""
def discount(r, bootstrap, size, gamma):
	R_batch = np.zeros([size], np.float64)
	R = bootstrap
	for i in reversed(range(0, size)):
		R = r[i] + gamma*R
		R_batch[i] = R
	return R_batch


"""
Calulates the Advantage Funtion for each timestep t in the batch, where:
	- At = \sum_{i=0}^{k-1} gamma^{i} * r_{t+i} + gamma^{k} * V(s_{t+k}) - V(s_{t})
where V(s_{t+k}) is the bootstraped value (if the episode hasn't finished).
This results in a 1-step update for timestep B, 2-step update for timestep
B-1,...., B-step discounted reward for timestep 1, where:
	- B = Batch Size
Example: consider B = 3. Therefore, it results in the following advantage vector:
 	- A[0] = r_1 + gamma * r_2 + gamma^2 * r_3 + gamma^3 * bootstrap - V(s_1)
 	- A[1] = r_2 + gamma * r_3 + gamma^2 * bootstrap - V(s_2)
 	- A[2] = r_3 + gamma * bootstrap - V(s_3)
"""
def calculate_advantage(r, v, bootstrap, size, gamma):
	A_batch = np.zeros([size], np.float64)
	aux = bootstrap
	for i in reversed(range(0, size)):
		aux = r[i] + gamma * aux
		A = aux - v[i]
		A_batch[i] = A
	return A_batch




class Worker():

	def __init__(self, num_id, env_id, gamma, global_episodes, model_path, render, save_img):
		self.name = "worker_" + str(num_id)
		"""
		Only worker 0 renders the game
		"""
		self.num_id = num_id
		self.increse_global_episode = global_episodes.assign_add(1)
		self.model_path = model_path

		"""
		Setup the environment:
			- we choose which environment we are going to use
			- we then retrieve some information based on the chosen environment
		"""
		self.env_id = env_id
		if self.env_id == ATARI:
			self.environment = Env_Atari('Pong-v0', render, num_id, save_img)
		else:
			self.environment = Env_Doom(render, num_id, save_img)
		self.num_actions = self.environment.get_num_action()
		self.height, self.width, self.channels = self.environment.get_state_space()

		"""
		Create the local network of the corresponding worker
		"""
		self.local_net = Network(self.name, self.num_actions, self.width, self.height, self.channels, gamma)
		self.update_local_net = self.local_net.update_network_op('worker_global')

	#-------------------------------------------------------------------
	def train(self, session, batch, bootstrap):
		"""
		Retrieve batch data
		"""
		batch = np.array(batch)
		inputs 	= batch[:,0]
		actions = batch[:,1]
		rewards = batch[:,2]
		values 	= batch[:,3]

		"""
		Reshape the 'states' array to fit the shape (B, height, width, channels)
		where B is the actual batch's size (B <= BATACH_SIZE) and
		B = len(states) = ... = len(values)
		"""
		states = []
		for s in inputs:
			states.append(s)
		states = np.array(states)

		"""
		Calculate the discounted rewards for each state in the batch
		"""
		R_batch = discount(rewards, bootstrap, len(rewards), self.local_net.gamma)

		"""
		Calculate the Advantage function for each step in the batch
		"""
		A_batch = calculate_advantage(rewards, values, bootstrap, len(rewards), self.local_net.gamma)

		"""
		Apply gradients w.r.t. the variables of the local network into the global network
		"""
		lstm_state = self.local_net.lstm_state_init
		grads, batch_loss = session.run([self.local_net.apply_grads, self.local_net.total_loss],
								feed_dict={	self.local_net.state_in[0]:lstm_state[0],
											self.local_net.state_in[1]:lstm_state[1],
											self.local_net.state	: states,
											self.local_net.R		: R_batch,
											self.local_net.actions	: actions,
											self.local_net.A		: A_batch 	})

		return batch_loss / len(batch)


	#-------------------------------------------------------------------
	def work(self, coordinator, session, saver):
		a_indexes = np.arange(self.num_actions)

		with session.as_default(), session.graph.as_default():
			while not coordinator.should_stop():
				"""
				Copies the master network to the local network
				"""
				session.run(self.update_local_net)

				batch_buffer = []
				episode_step_count = 0
				episode_batches = 0
				episode_mean_loss = 0
				episode_reward = 0
				done = False
				actions_chosen = np.zeros((self.num_actions), np.int64)

				"""
				Resets the environment and the LSTM state
				"""
				s = self.environment.reset_environment()
				lstm_state = self.local_net.lstm_state_init

				"""
				Loop for each episode
				"""
				while done == False and episode_step_count < MAX_EPISODE_LENGTH:
					"""
					Retrieve the the value fnction and the action probabilities
					of state s (pi(s) and V(s)), whose shapes are:
						- pi(s) = (?, NUM_ACTIONS)
						- V(s) = (?, 1)
					Where ? is the batch_size. In the following case,
					batch_size = 1, so we reshape both outputs to:
						- pi(s) = (NUM_ACTIONS)
						- V(s) = (1)
					"""
					#action_prob_batch, v_batch, lstm_state = session.run([self.local_net.policy, self.local_net.value, self.local_net.state_out],
					v_batch, a, lstm_state = session.run([self.local_net.value, self.local_net.action, self.local_net.state_out],
																		feed_dict={	self.local_net.state:[s],
																					self.local_net.state_in[0]:lstm_state[0],
				                            										self.local_net.state_in[1]:lstm_state[1]})
					#action_prob = action_prob_batch[0]
					v = v_batch[0][0]

					"""
					Select a random action based on the action's probability distribution
					"""
					#a = np.random.choice(a_indexes, p=action_prob)
					actions_chosen[a] += 1

					"""
					Perform the action
					"""
					s1, r, done = self.environment.perform_action(a)
					if done == True:
						s1 = s

					"""
					Insert data into the batch buffer
					"""
					batch_buffer.append([s,a,r,v])
					episode_reward += r

					"""
					Update the master network if the batch buffer is full
					"""
					if len(batch_buffer) >= BATCH_SIZE and done == False:
						"""
						Since we don't know the final return, we approximate
						it by the V(s1)
						"""
						v1_batch = session.run(self.local_net.value,
											feed_dict={	self.local_net.state:[s1],
														self.local_net.state_in[0]:lstm_state[0],
														self.local_net.state_in[1]:lstm_state[1]})
						v1 = v1_batch[0][0]
						batch_loss = self.train(session, batch_buffer, v1)
						episode_mean_loss += batch_loss
						episode_batches += 1
						batch_buffer = []
						session.run(self.update_local_net)



					s = s1
					episode_step_count += 1
					if episode_step_count >= MAX_EPISODE_LENGTH:
						print "\t\tMAX EPISODE LENGTH reached! (",self.name,")"

					if done == True:
						if episode_step_count < BATCH_SIZE:
							print "\t\tEpisode finished before filling a batch"
						break

				if len(batch_buffer) > 0:
					batch_loss = self.train(session, batch_buffer, 0)
					episode_mean_loss += batch_loss
					episode_batches += 1

				global_count = session.run(self.increse_global_episode)

				print "GC = ", global_count,"\tEpisode Reward = ", episode_reward,"\tBatch Loss = ",episode_mean_loss / episode_batches, "\tEpisodes Steps = ",episode_step_count,"\tActions = ", actions_chosen, "\t(", self.name, ")"

				if global_count % SAVER_INTERVAL == 0:
					print "Saving model.............."
					saver.save(session,self.model_path+'/model-'+str(global_count)+'.cptk')
					print "Model saved!"
