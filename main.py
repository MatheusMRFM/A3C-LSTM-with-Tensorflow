from time import sleep
from time import time
import numpy as np
import tensorflow as tf
import time, random, threading
import multiprocessing
from Network import *
from Worker import *
from Summary import *

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
num_workers = 4		#multiprocessing.cpu_count()
model_path = './model'

tf.reset_default_graph()

"""
Creates the master worker that maintains the master network.
We then initialize the workers array.
"""
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
total_frames = tf.Variable(0,dtype=tf.int32,name='total_frames',trainable=False)
learning_rate = tf.train.polynomial_decay(LEARNING_RATE,total_frames,MAX_ITERATION//2,
										  LEARNING_RATE*0.1)

with tf.device("/cpu:0"):
	summary_writer = tf.summary.FileWriter("./train/"+SUMMARY_NAME)
	summary = Summary(summary_writer)
	master_worker = Worker('global', env, GAMMA, learning_rate, global_episodes, total_frames, model_path, False, False, num_workers, summary)
	workers = []
	for i in range(num_workers):
		print (i)
		workers.append(Worker(i, env, GAMMA, learning_rate, global_episodes, total_frames, model_path, render, save_img, num_workers, summary))

"""
Initializes tensorflow variables
"""
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1)) as session:
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
