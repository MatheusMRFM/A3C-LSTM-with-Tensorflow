from Environment import *
import gym
import multiprocessing, threading
import scipy.misc
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imsave
from scipy import misc

MAX_EPISODE_LENGTH = 80000000
lock = threading.Lock()



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
		s = self.process_image(s)
		self.current_state = np.stack((s, s, s, s), axis = 2)

		return self.current_state

	#-------------------------------------------------------------------
	def get_state_space(self):
		return self.height, self.width

	#-------------------------------------------------------------------
	def get_num_action(self):
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

		s1, r, d, i = self.env.step(action)
		s1 = self.process_image(s1)

		s1_expanded = np.expand_dims(s1, axis=2)
		self.current_state = np.append(self.current_state[:,:,1:], s1_expanded, axis = 2)

		self.finished = d
		self.info = i
		if self.save_img == True:
			self.save_image(s1)

		return self.current_state, r, d

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
		s = image[34:34+160, :160]
		#s = resize(rgb2gray(image), (2*self.height, 2*self.width))
		#s = resize(s, (self.height, self.width))
		#return np.expand_dims(s, axis=2)
		#s = misc.imresize(s, (2*self.height, 2*self.width))
		s = misc.imresize(s, (self.height, self.width))
		s = s.mean(2)
		s = s.astype(np.float32)
		s *= (1.0 / 255.0)
		return s

	#-------------------------------------------------------------------
	def help_message(self):
		pass
