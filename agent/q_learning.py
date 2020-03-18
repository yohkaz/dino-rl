import numpy as np
from agent.agent import Agent

class QLearningAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.qtable = np.zeros((env.observation_space.high[0], env.observation_space.high[1], env.action_space.n))
        print("QLearningAgent:")
        super().print_commands()

    def update_qtable(self, state, action, new_state, next_action, reward):
        cell = state + (action,)
        td = reward + self.parameters['gamma'] * np.max(self.qtable[new_state]) - self.qtable[cell]
        self.qtable[cell] = self.qtable[cell] + self.parameters['lr'] * td
