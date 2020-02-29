import base64
import io
import numpy as np
import os
from collections import deque
from PIL import Image               # Load images

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from game import DinoGame

class DinoEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, simplified_state=True, render=True, accelerate=True):
        self.game = DinoGame(render)

        # image_size = self._observe().shape
        # self.observation_space = spaces.Box(low=0, high=255, shape=(150, 600, 3), dtype=np.uint8)

        if simplified_state:
            self.observation_space = spaces.Discrete(21)
        self.action_space = spaces.Discrete(3)
        self.gametime_reward = 0.1
        self.gameover_penalty = -1
        # self.current_frame = self.observation_space.low
        self._action_set = [0, 1, 2]

    def process_img(self, img):
        width, height = img.size

        # Remove the character, ground, and score txt
        (left, upper, right, lower) = (42, 22, width, height-18)
        img = img.crop((left, upper, right, lower))

        # Find first obstacle
        n_level = 20
        state = {'dist_obstacle': n_level}
        width, height = img.size
        # obstacle = False
        for i in range(width):
            coord = (i, height-20)
            if img.getpixel(coord) != (255, 255, 255):
                # img.putpixel(coord, (255, 0, 0))
                # obstacle = True
                state['dist_obstacle'] = int(((i - (i % 10)) * n_level) / (width - (width % 10)))
                break

        # if obstacle:
        #     print(state)

        # return state
        return state['dist_obstacle']

    def _observe(self):
        # TODO: define state as intended
        s = self.game.get_canvas()
        b = io.BytesIO(base64.b64decode(s))
        i = Image.open(b)

        # RGBa to RGB, needed ?
        bg = Image.new("RGB", i.size, (255, 255, 255))  # fill background as white color
        bg.paste(i, mask=i.split()[3])  # 3 is the alpha channel
        i = bg

        state = self.process_img(i.copy())

        a = np.array(i)
        self.current_frame = a
        # return self.current_frame
        return state

    def resume(self):
        self.game.resume()

    def step(self, action):
        if action == 1:
            self.game.press_up()
        if action == 2:
            self.game.press_down()
        if action == 3:
            self.game.press_space()
        # observation = int((self._observe() + self._observe() + self._observe() + self._observe()) / 4)
        observation = self._observe()
        reward = self.gametime_reward
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

    def render(self):
        s = self.game.get_canvas()
        b = io.BytesIO(base64.b64decode(s))
        i = Image.open(b)

        # RGBa to RGB, needed ?
        bg = Image.new("RGB", i.size, (255, 255, 255))  # fill background as white color
        bg.paste(i, mask=i.split()[3])  # 3 is the alpha channel
        bg.show()

    def close(self):
        self.game.close()

    def get_score(self):
        # print(self.game.is_playing())
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
