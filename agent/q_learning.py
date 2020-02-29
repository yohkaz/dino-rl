import numpy as np
import random

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.qtable = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state, epsilon):
        # Choose an action a in the current world state (s)
        ## First we randomize a number
        explore_exploit = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if explore_exploit > epsilon:
            action = np.argmax(self.qtable[state, :])
        # Else doing a random choice --> exploration
        else:
            action = self.env.action_space.sample()

        return action

    def train(self, n_episodes, parameters):
        self.rewards = []

        epsilon = parameters['max_epsilon']
        for episode in range(n_episodes):
            print("****************************************************")
            print("TRAIN EPISODE", episode)
            print("epsilon:", epsilon)

            # Reset the environment
            state = self.env.reset()
            step = 0
            done = False
            total_rewards = 0

            while True:
                # print(state, "", end='', flush=True)
                action = self.choose_action(state, epsilon)

                # print("action:", self.env.get_action_meanings()[action], "", end="", flush=True)
                print("action:", self.env.get_action_meanings()[action])
                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done, info = self.env.step(action)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                # qtable[new_state,:] : all the actions we can take from new state
                self.qtable[state, action] = self.qtable[state, action] + parameters['lr'] * (reward + parameters['gamma'] * np.max(self.qtable[new_state, :]) - self.qtable[state, action])

                total_rewards += reward

                # Our new state is state
                state = new_state

                # If done (if we're dead) : finish episode
                if done == True: 
                    break

            print()
            # Reduce epsilon (because we need less and less exploration)
            epsilon = parameters['min_epsilon'] + (parameters['max_epsilon'] - parameters['min_epsilon'])*np.exp(-parameters['decay_rate']*episode) 
            self.rewards.append(total_rewards)

    def run(self, n_episodes):
        self.env.reset()

        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            print("****************************************************")
            print("RUN EPISODE", episode)

            while True:
                # Take the action (index) that have the maximum expected future reward given that state
                action = np.argmax(self.qtable[state,:])
                new_state, reward, done, info = self.env.step(action)

                if done:
                    # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
                    self.env.render()
                    print("Score:", self.env.get_score())
                    break
                state = new_state

    def reset(self):
        self.env.__init__()
        self.qtable = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def print_infos(self):
        print("Reward:")
        print(self.rewards)
        print("Q-table:")
        print(self.qtable)
