from Environment import *
import multiprocessing, threading
from vizdoom import *
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.io import imsave

ACTION_SIZE = 3

class Env_Doom(Environment):

	def __init__(self, render, worker_id, save_img):
		self.height = RESEIZE_HEIGHT
		self.width = RESEIZE_WIDTH
		self.channels = CHANNELS
		self.save_img = save_img
		self.render = render
		self.env_id = worker_id
		self.num_img = 0

		self.env = DoomGame()
		self.env.set_doom_scenario_path("basic.wad") #This corresponds to the simple task we will pose our agent
		self.env.set_doom_map("map01")
		self.env.set_screen_resolution(ScreenResolution.RES_160X120)
		self.env.set_screen_format(ScreenFormat.GRAY8)
		self.env.set_render_hud(render)
		self.env.set_render_crosshair(render)
		self.env.set_render_weapon(render)
		self.env.set_render_decals(render)
		self.env.set_render_particles(render)
		self.env.add_available_button(Button.MOVE_LEFT)
		self.env.add_available_button(Button.MOVE_RIGHT)
		self.env.add_available_button(Button.ATTACK)
		self.env.add_available_game_variable(GameVariable.AMMO2)
		self.env.add_available_game_variable(GameVariable.POSITION_X)
		self.env.add_available_game_variable(GameVariable.POSITION_Y)
		self.env.set_episode_timeout(300)
		self.env.set_episode_start_time(10)
		self.env.set_window_visible(render)
		self.env.set_sound_enabled(False)
		self.env.set_living_reward(-1)
		self.env.set_mode(Mode.PLAYER)
		self.env.init()
		self.actions = np.identity(ACTION_SIZE, dtype=bool).tolist()
		self.reset_environment()

	#-------------------------------------------------------------------
	def reset_environment(self):
		self.env.new_episode()
		s = self.env.get_state().screen_buffer
		self.current_state = self.process_image(s)
		return self.current_state

	#-------------------------------------------------------------------
	def get_state_space(self):
		return [self.height, self.width, self.channels]

	#-------------------------------------------------------------------
	def get_num_action(self):
		return ACTION_SIZE

	#-------------------------------------------------------------------
	def get_current_state(self):
		return self.current_state

	#-------------------------------------------------------------------
	def perform_action(self, action):
		r = self.env.make_action(self.actions[action]) / 100.0
		d = self.env.is_episode_finished()
		if d == False:
			s = self.env.get_state().screen_buffer
			self.current_state = self.process_image(s)

		self.finished = d

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
		s = resize(image, (self.height, self.width))
		return np.expand_dims(s, axis=2)

	#-------------------------------------------------------------------
	def help_message(self):
		pass
