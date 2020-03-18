from gym_env import DinoEnv
from agent import *
from agent.agent import EPSILON, SOFTMAX

agent = None
parameters = None

def simplified_state_best_experiment():
    global agent
    global parameters

    agent = SarsaAgent(DinoEnv(simplified_state=True, accelerate=True))

    parameters = {
        'policy': EPSILON,
        'epsilon': 0.5,
        'epsilon_decay': 0.999,
        'tau': 1,
        'tau_inc': 0.001,
        'lr': 0.05,
        'gamma': 0.95,
    }

    agent.set_parameters(parameters)
    agent.train(150)
    agent.print_infos()

def deep_experiment_2():
    global agent
    global parameters

    # Accelerate OFF
    agent = DeepQLAgent(DinoEnv(simplified_state=False, accelerate=False))

    parameters = {
        'policy': EPSILON,
        'epsilon': 0.9,
        'epsilon_decay': 0.999,
        'gamma': 0.95,
    }

    agent.set_parameters(parameters)
    agent.train(200)
    agent.print_infos()

def deep_experiment_3():
    global agent
    global parameters

    # Accelerate ON
    agent = DeepQLAgent(DinoEnv(simplified_state=False, accelerate=True))

    parameters = {
        'policy': EPSILON,
        'epsilon': 0.9,
        'epsilon_decay': 0.999,
        'gamma': 0.95,
    }

    agent.set_parameters(parameters)
    agent.train(1500)
    agent.print_infos()

def main():
    global agent
    global parameters

    # agent = QLearningAgent(DinoEnv(simplified_state=True, accelerate=False))
    # agent = SarsaAgent(DinoEnv(simplified_state=True, accelerate=False))
    agent = DeepQLAgent(DinoEnv(simplified_state=False, accelerate=True))

    parameters = {
        'policy': EPSILON,
        'epsilon': 0.9,
        'epsilon_decay': 0.999,
        'tau': 1,
        'tau_inc': 0.001,
        # 'lr': 0.15,
        'lr': 0.05,
        'gamma': 0.95,
        # 'gamma': 0.4,
    }

    agent.set_parameters(parameters)

if  __name__ == "__main__":
    # run with 'python -i main.py' to interact with the agent in the command line
    main()
