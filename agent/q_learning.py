import numpy as np
from agent.agent import Agent

class QLearningAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        print("QLearningAgent:")
        super().print_commands()

    def update_qtable(self, state, action, new_state, next_action, reward):
        # td = reward + self.parameters['gamma'] * np.max(self.qtable[new_state, :]) - self.qtable[state, action]
        # self.qtable[state, action] = self.qtable[state, action] + self.parameters['lr'] * td
        cell = state + (action,)
        td = reward + self.parameters['gamma'] * np.max(self.qtable[new_state]) - self.qtable[cell]
        self.qtable[cell] = self.qtable[cell] + self.parameters['lr'] * td
