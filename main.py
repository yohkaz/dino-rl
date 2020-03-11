from gym_env import DinoEnv
from agent import *
from agent.agent import EPSILON, SOFTMAX

agent = None
parameters = None

def main():
    global agent
    global parameters

    # agent = QLearningAgent(DinoEnv(accelerate=True), FrameProcessor(simplified=True))
    agent = SarsaAgent(DinoEnv(simplified_state=True, accelerate=True))

    parameters = {
        'policy': EPSILON,
        'epsilon': 0.5,
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
