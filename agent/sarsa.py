import numpy as np
from agent.agent import Agent, SOFTMAX

class SarsaAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.qtable = np.zeros((env.observation_space.high[0], env.observation_space.high[1], env.action_space.n))
        print("SarsaAgent:")
        super().print_commands()

    def update_qtable(self, state, action, new_state, next_action, reward):
        cell = state + (action,)

        if self.parameters['policy'] == SOFTMAX:
            prob_a = self.softmax(self.parameters['tau'], self.qtable[new_state])
            next_value = np.sum(prob_a * self.qtable[new_state])
        else:
            next_cell = new_state + (next_action,)
            next_value = self.qtable[next_cell]

        td = reward + self.parameters['gamma'] * next_value - self.qtable[cell]
        self.qtable[cell] = self.qtable[cell] + self.parameters['lr'] * td
