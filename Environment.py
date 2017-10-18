import numpy as np
import time, random
from abc import ABCMeta, abstractmethod

RESEIZE_HEIGHT = 84
RESEIZE_WIDTH = 84
CHANNELS = 1

class Environment():
	__metaclass__ = ABCMeta

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def reset_environment(self):
		pass

	@abstractmethod
	def get_state_space(self):
		pass

	@abstractmethod
	def get_num_action(self):
		pass

	@abstractmethod
	def get_current_state(self):
		pass

	@abstractmethod
	def perform_action(self, action):
		pass

	@abstractmethod
	def help_message(self):
		pass
