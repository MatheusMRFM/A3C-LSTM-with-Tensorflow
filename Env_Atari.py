from Environment import *
import gym
import multiprocessing, threading
import scipy.misc
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imsave

MAX_EPISODE_LENGTH = 80000000
lock = threading.Lock()

RESEIZE_HEIGHT = 47
RESEIZE_WIDTH = 47
CHANNELS = 1



class Env_Atari(Environment):

	def __init__(self, game_name, render, worker_id, save_img):
		self.num_img = 0
		self.env_id = worker_id
		self.save_img = save_img

		self.env = gym.make(game_name)
		self.render = render
		#self.height = self.env.observation_space.shape[0]
		#self.width = self.env.observation_space.shape[1]
		#self.channels = self.env.observation_space.shape[2]
		self.height = RESEIZE_HEIGHT
		self.width = RESEIZE_WIDTH
		self.channels = CHANNELS
		self.reset_environment()
		self.finished = False

	#-------------------------------------------------------------------
	def reset_environment(self):
		"""
		self.current_state is a 3D matrix with the shape
		returned by self.get_state_space()
		"""
		s = self.env.reset()
		self.current_state = self.process_image(s)

		return self.current_state

	#-------------------------------------------------------------------
	def get_state_space(self):
		return [self.height, self.width, self.channels]

	#-------------------------------------------------------------------
	def get_num_action(self):
		#if (self.env.spec.id == "Pong-v0" or self.env.spec.id == "Breakout-v0"):
		#	return 3
		return self.env.action_space.n

	#-------------------------------------------------------------------
	def get_current_state(self):
		return self.current_state

	#-------------------------------------------------------------------
	def perform_action(self, action):
		if self.render == True:
			lock.acquire()
			self.env.render()
			lock.release()
		"""
		If the game is Pong or Breakout, the valid actions are 1, 2, 3, that is,
		action 0 is removed. Therefore, we add 1 to the current action to make it valid
		"""
		#if (self.env.spec.id == "Pong-v0" or self.env.spec.id == "Breakout-v0"):
		#	a = action + 1
		#else:
		a = action


		s1, r, d, i = self.env.step(a)
		self.current_state = self.process_image(s1)

		self.finished = d
		self.info = i
		if self.save_img == True:
			self.save_image(self.current_state)

		return [self.current_state, r, d]

	#-------------------------------------------------------------------
	def save_image(self, image):
		"""
		Save game images for debugging (if necessary). To save images, create
		a folder named "img/" and, inside it, create one folder for each thread,
		each one named thread.id (0, 1, 2, ...)
		"""
		name = "img/" + str(self.env_id) + "/img" + str(self.num_img) + ".png"
		imsave(name, image)
		self.num_img += 1

	#-------------------------------------------------------------------
	def process_image (self, image):
		"""
		Transform the image into grayscale and resizes it
		"""
		s = resize(rgb2gray(image), (self.height, self.width))
		return np.expand_dims(s, axis=2)

	#-------------------------------------------------------------------
	def help_message(self):
		pass
