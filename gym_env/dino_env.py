import base64
import io
import numpy as np
import os
from PIL import Image               # Load images

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from game import DinoGame
from gym_env.frame_processor import FrameProcessor

class DinoEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, simplified_state=True, accelerate=True):
        self.game = DinoGame(accelerate)
        self.frame_processor = FrameProcessor(simplified=simplified_state)

        # Set env parameters
        self.accelerate = accelerate
        if simplified_state:
            self.observation_space = spaces.Box(low=np.array([0, 0]),
                                                high=np.array([self.frame_processor.dimensions()[0],
                                                               self.frame_processor.dimensions()[1]]),
                                                               dtype=np.int)
        self.action_space = spaces.Discrete(3)
        self.gametime_reward = 1
        self.gameover_penalty = -20
        self._action_set = [0, 1, 2]

    def _observe(self):
        s = self.game.get_canvas()
        b = io.BytesIO(base64.b64decode(s))
        i = Image.open(b)

        # RGBa to RGB
        bg = Image.new("RGB", i.size, (255, 255, 255))  # fill background as white color
        bg.paste(i, mask=i.split()[3])  # 3 is the alpha channel
        i = bg

        self.current_frame = i
        return self.frame_processor.process(self.current_frame)

    def resume(self):
        self.game.resume()

    def step(self, action):
        reward = 0
        if action == 1:
            self.game.press_up()
            reward = -1
        if action == 2:
            self.game.press_down()
            reward = -1
        if action == 3:
            self.game.press_space()

        observation = self._observe()
        reward = reward + self.gametime_reward
        done = False
        info = {}
        if self.game.is_crashed():
            reward = self.gameover_penalty
            done = True
        elif not self.game.is_playing():
            self.game.resume()

        return observation, reward, done, info

    def reset(self, record=False):
        self.game.restart()
        return self._observe()

    def relaunch_game(self):
        self.game = DinoGame(self.accelerate)

    def render(self):
        s = self.game.get_canvas()
        b = io.BytesIO(base64.b64decode(s))
        i = Image.open(b)

        # RGBa to RGB, needed ?
        bg = Image.new("RGB", i.size, (255, 255, 255))  # fill background as white color
        bg.paste(i, mask=i.split()[3])  # 3 is the alpha channel
        bg.show()

    def show_current_frame(self):
        self.current_frame.show()
        return self.current_frame

    def close(self):
        self.game.close()

    def get_score(self):
        return self.game.get_score()

    def set_acceleration(self, enable):
        if enable:
            self.game.restore_parameter('config.ACCELERATION')
        else:
            self.game.set_parameter('config.ACCELERATION', 0)

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

ACTION_MEANING = {
    0 : "NOOP",
    1 : "UP",
    2 : "DOWN",
    3 : "SPACE",
}
