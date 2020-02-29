from gym_env import DinoEnv
from agent import *

# dinoEnv = DinoEnv(render=True)

# dinoEnv.step(3)
# dinoEnv.resume()

# i = 1
# while True:
#     # if i % 200 == 0:
#     observation, reward, done, info = dinoEnv.step(1)
#     if done:
#         print("Done!")
#         # break
#         dinoEnv.reset()

#     # print(dinoEnv.get_score())
#     i += 1
agent = None
parameters = None

def qlearning():
    global agent
    global parameters
    agent = QLearningAgent(DinoEnv(render=True))

    parameters = {
        'lr': 0.7,
        'gamma': 0.9,
        'max_epsilon': 0.8,
        'min_epsilon': 0.01,
        'decay_rate': 0.005
    }

    agent.train(200, parameters)
    agent.print_infos()
    agent.run(5)

def sarsa():
    global agent
    global parameters
    agent = SarsaAgent(DinoEnv(render=True))

    parameters = {
        'lr': 0.7,
        'gamma': 0.9,
        'max_epsilon': 0.5,
        'min_epsilon': 0.01,
        'decay_rate': 0.0005
    }

    agent.train(200, parameters)
    # agent.print_infos()
    # agent.run(5)

if  __name__ == "__main__":
    # qlearning()
    sarsa()
