import numpy as np
from agent.agent import Agent, SOFTMAX

class SarsaAgent(Agent):
    def __init__(self, env):
        super().__init__(env)
        print("SarsaAgent:")
        super().print_commands()

    def update_qtable(self, state, action, new_state, next_action, reward):
        # td = reward + self.parameters['gamma'] * self.qtable[new_state, next_action] - self.qtable[state, action]
        # self.qtable[state, action] = self.qtable[state, action] + self.parameters['lr'] * td
        cell = state + (action,)

        if self.parameters['policy'] == SOFTMAX:
            prob_a = self.softmax(self.parameters['tau'], self.qtable[new_state])
            next_value = np.sum(prob_a * self.qtable[new_state])
            # print(prob_a)
            # print(self.qtable[new_state])
            # print(prob_a * self.qtable[new_state])
            # print(next_value)
        else:
            next_cell = new_state + (next_action,)
            next_value = self.qtable[next_cell]

        td = reward + self.parameters['gamma'] * next_value - self.qtable[cell]
        self.qtable[cell] = self.qtable[cell] + self.parameters['lr'] * td
