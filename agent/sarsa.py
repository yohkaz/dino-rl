import numpy as np
from agent.agent import Agent

class SarsaAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        print("SarsaAgent:")
        super().print_commands()

    def update_qtable(self, state, action, new_state, next_action, reward):
        td = reward + self.gamma * self.qtable[new_state, next_action] - self.qtable[state, action]
        self.qtable[state, action] = self.qtable[state, action] + self.lr * td
