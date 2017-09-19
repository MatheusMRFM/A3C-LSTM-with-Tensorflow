from time import sleep
from time import time
import numpy as np
import tensorflow as tf
import time, random, threading
import multiprocessing
from Network import *
from Worker import *

"""
- 	To load a model located in 'model_path', just set load = True
- 	To render the environment, set render = True
- 	To save the images that are fed to the network, set save_img = True. For
	more info in this option, check the "save_image" method of the Env_Atari class
"""
load = False
render = False
save_img = False
env = ATARI
num_workers = 8		#multiprocessing.cpu_count()
model_path = './model'

tf.reset_default_graph()

"""
Creates the master worker that maintains the master network.
We then initialize the workers array.
"""
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
with tf.device("/cpu:0"):
	master_worker = Worker('global', env, GAMMA, global_episodes, model_path, False, False)
	workers = []
	for i in range(num_workers):
		print (i)
		workers.append(Worker(i, env, GAMMA, global_episodes, model_path, render, save_img))

"""
Initializes tensorflow variables
"""
with tf.Session() as session:
	saver = tf.train.Saver(max_to_keep=5)
	if load:
		print ("Loading....")
		c = tf.train.get_checkpoint_state(model_path)
		saver.restore(session,c.model_checkpoint_path)
		print ("Graph loaded!")
	else:
		session.run(tf.global_variables_initializer())
	coord = tf.train.Coordinator()

	"""
	Initializes the worker threads
	"""
	worker_threads = []
	for i in range(num_workers):
		t = threading.Thread(target=workers[i].work, args=(coord, session, saver))
		t.start()
		sleep(0.5)
		worker_threads.append(t)

	coord.join(worker_threads)
