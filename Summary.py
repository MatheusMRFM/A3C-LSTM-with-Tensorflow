import numpy as np
import tensorflow as tf

SUMMARY_INTERVAL = 4
FRAMES_IN_EPOCH = 4000000
SUMMARY_NAME = "A3C_"

class Summary():
    def __init__(self, writer):
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_values = []
        self.last_write = 0
        self.writer = writer

    def add_info(self, r, l, mean_v, frame_count, global_count):
        self.ep_rewards.append(r)
        self.ep_lengths.append(l)
        self.ep_values.append(mean_v)
        if global_count - self.last_write >= SUMMARY_INTERVAL:
            self.last_write = global_count
            self.write(frame_count)

    def reset(self):
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_values = []

    def write(self, frame_count):
        mean_reward = np.mean(self.ep_rewards)
        mean_length = np.mean(self.ep_lengths)
        mean_value = np.mean(self.ep_values)
        summary = tf.Summary()
        summary.value.add(tag=SUMMARY_NAME+'/Reward', simple_value=float(mean_reward))
        summary.value.add(tag=SUMMARY_NAME+'/Length', simple_value=float(mean_length))
        summary.value.add(tag=SUMMARY_NAME+'/Value', simple_value=float(mean_value))
        #summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
        #summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
        #summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
        #summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
        #summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
        #count = float(frame_count) / float(FRAMES_IN_EPOCH)
        self.writer.add_summary(summary, frame_count)
        self.writer.flush()
        self.reset()
