from gym_env import DinoEnv
from agent import *
from agent.agent import EPSILON, SOFTMAX

agent = None
parameters = None

def qlearning():
    global agent
    global parameters
    agent = QLearningAgent(DinoEnv(render=True))

    # parameters = {
    #     'lr': 0.7,
    #     'gamma': 0.9,
    #     'max_epsilon': 0.5,
    #     'min_epsilon': 0.01,
    #     'decay_rate': 0.0
    # }

    parameters = {
        'mode': EPSILON,
        'init_epsilon': 0.5,
        'epsilon_decay': 0.999,
        'lr': 0.7,
        'gamma': 0.9,
    }

    # agent.train(100, parameters)
    # agent.print_infos()
    # agent.run(5)

def sarsa():
    global agent
    global parameters
    agent = SarsaAgent(DinoEnv(render=True))

    parameters = {
        'mode': EPSILON,
        'init_epsilon': 0.5,
        'epsilon_decay': 0.999,
        'lr': 0.7,
        'gamma': 0.9,
    }

    agent.set_parameters(parameters)
    # agent.train(400, parameters)
    # agent.train(1, parameters)
    # agent.print_infos()
    # agent.run(5)

if  __name__ == "__main__":
    # qlearning()
    sarsa()
