import numpy as np
import tensorflow as tf
import time, random, threading
import multiprocessing
import scipy.misc
from Network import *
#from Env_Doom import *
from Env_Atari import *

MAX_ITERATION = 500000000

SAVER_INTERVAL 	= 100

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

#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
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

#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
class Batch():
	def __init__(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.values = []
		self.bootstrap = 0.0
		self.initial_feature = None
		self.size = 0

	def add_data(self, s, a, r, v):
		self.states.append(s)
		self.actions.append(a)
		self.rewards.append(r)
		self.values.append(v)
		self.size += 1

	def reset(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.values = []
		self.bootstrap = 0.0
		self.initial_feature = None
		self.size = 0

#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
class Worker():

	def __init__(self, num_id, env_id, gamma, learning_rate, global_episodes, total_frames, model_path, render, save_img, num_worker, summary):
		self.num_worker = num_worker
		self.name = "worker_" + str(num_id)
		self.summary = summary
		self.global_episodes = global_episodes
		self.num_id = num_id
		self.increse_global_episode = global_episodes.assign_add(1)
		self.increase_frames = total_frames.assign_add(1)
		self.total_frames = total_frames
		self.model_path = model_path

		"""
		Setup the environment:
			- we choose which environment we are going to use
			- we then retrieve some information based on the chosen environment
		"""
		self.env_id = env_id
		if self.env_id == ATARI:
			self.environment = Env_Atari('PongDeterministic-v4', render, num_id, save_img)
		else:
			self.environment = Env_Doom(render, num_id, save_img)
		self.num_actions = self.environment.get_num_action()
		self.height, self.width = self.environment.get_state_space()

		"""
		Create the local network of the corresponding worker
		"""
		self.local_net = Network(self.name, self.num_actions, self.width, self.height, 0, gamma, learning_rate)
		self.update_local_net = self.local_net.update_network_op('worker_global')

	#-------------------------------------------------------------------
	def train(self, session, batch):
		"""
		Clip reward to [-1, 1]
		"""
		batch.rewards = np.array(batch.rewards)
		np.clip(batch.rewards, -1.0, 1.0, out=batch.rewards)

		"""
		Calculate the discounted rewards for each state in the batch
		"""
		R_batch = discount(batch.rewards, batch.bootstrap, batch.size, self.local_net.gamma)

		"""
		Calculate the Advantage function for each step in the batch
		"""
		A_batch = calculate_advantage(batch.rewards, batch.values, batch.bootstrap, batch.size, self.local_net.gamma)

		"""
		Retrieve only the LSTM state that occured at the begining of the current
		batch, that is, the first LSTM state of the batch. The remaining features
		in the batch are discarted
		"""
		lstm_state = batch.initial_feature

		"""
		Apply gradients w.r.t. the variables of the local network into the global network
		"""
		_, batch_loss = session.run([self.local_net.apply_grads, self.local_net.total_loss],
								feed_dict={	self.local_net.state_in[0]:lstm_state[0],
											self.local_net.state_in[1]:lstm_state[1],
											self.local_net.state	: batch.states,
											self.local_net.R		: R_batch,
											self.local_net.actions	: batch.actions,
											self.local_net.A		: A_batch 	})

		return batch_loss / batch.size


	#-------------------------------------------------------------------
	def work(self, coordinator, session, saver):
		a_indexes = np.arange(self.num_actions)
		done = True
		batch = Batch()
		global_count = -1

		"""
		Main loop
		"""
		while not coordinator.should_stop():
			"""
			New episode starting
			"""
			if done:
				"""
				Resets the environment and the LSTM state
				"""
				s = self.environment.reset_environment()
				last_lstm_state = self.local_net.lstm_state_init

				done = False
				rewards = 0
				length = 0
				total_values = 0
				actions_chosen = np.zeros((self.num_actions), np.int64)

			"""
			Copy weights from global to local network and
			save the first LSTM state of the current batch
			"""
			session.run(self.update_local_net)
			batch.initial_feature = last_lstm_state

			"""
			Obtain BATCH_SIZE experiences
			"""
			for _ in range(0, BATCH_SIZE):
				pi, v_batch, last_lstm_state = session.run([self.local_net.policy, self.local_net.value, self.local_net.state_out],
																	feed_dict={	self.local_net.state:[s],
																				self.local_net.state_in[0]:last_lstm_state[0],
			                            										self.local_net.state_in[1]:last_lstm_state[1]})
				v = v_batch[0][0]
				a = np.random.choice(a_indexes, p=pi[0])
				actions_chosen[a] += 1

				"""
				Perform the action and then insert data into the batch
				"""
				s1, r, done = self.environment.perform_action(a)
				batch.add_data(s,a,r,v)
				s = s1

				"""
				Update episode info (used for the summary)
				"""
				rewards += r
				length += 1
				total_values += v
				frame_count = session.run(self.increase_frames)

				if done:
					global_count = session.run(self.increse_global_episode)
					print "GC = ", global_count,"\tEpisode Reward = ", rewards ,"\tEpisodes Steps = ",length,"\tActions = ", actions_chosen, "\tFrames = ", frame_count
					break

			"""
			Update the master network, since the batch buffer is already full
			"""
			if not done:
				v1_batch = session.run(self.local_net.value,
									feed_dict={	self.local_net.state:[s],
												self.local_net.state_in[0]:last_lstm_state[0],
												self.local_net.state_in[1]:last_lstm_state[1]})
				batch.bootstrap = v1_batch[0][0]
			else:
				batch.bootstrap = 0.0
			batch_loss = self.train(session, batch)

			"""
			Update the Summary
			"""
			if done:
				self.summary.add_info(rewards, length, total_values/length, frame_count, global_count)

			"""
			Reset the batch
			"""
			batch.reset()

			"""
			Save the model
			"""
			if done and global_count % SAVER_INTERVAL == 0:
				print ("Saving model..............")
				saver.save(session,self.model_path+'/model-'+str(global_count)+'.cptk')
				print ("Model saved!")
