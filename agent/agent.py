import numpy as np
import random

# Action modes
EPSILON = "EPSILON"
SOFTMAX = "SOFTMAX"

class Agent:
    def __init__(self, env):
        self.env = env
        self.qtable = np.zeros((env.observation_space.n, env.action_space.n))

    def set_parameters(self, parameters):
        self.mode = parameters['mode']
        self.epsilon = parameters['init_epsilon']
        self.epsilon_decay = parameters['epsilon_decay']
        if self.mode == SOFTMAX:
            self.tau = parameters['init_tau']
            self.tau_inc = parameters['tau_inc']
        self.lr = parameters['lr']
        self.gamma = parameters['gamma']

    def choose_action(self, state):
        action = None
        if self.mode == EPSILON:
            if random.uniform(0, 1) > self.epsilon:
                action = np.argmax(self.qtable[state, :])
            else:
                # random choice --> exploration
                action = self.env.action_space.sample()
        elif self.mode == SOFTMAX:
            # Compute probabilities for each action
            prob_a = softmax(self.tau, self.qtable[state, :])
            # Cumulative sum
            cumsum_a = np.cumsum(prob_a)
            # Pick an action i with probability prob_a[i]
            action = np.where(np.random.rand() < cumsum_a)[0][0]
        else:
            raise ValueError("Wrong Action Mode !")

        print(self.env.get_action_meanings()[action], end=" ", flush=True)
        return action

    def update_qtable(self, state, action, new_state, next_action, reward):
        raise NotImplementedError("Must override update_qtable")

    def train(self, n_episodes):
        self.rewards = []

        for episode in range(n_episodes):
            print("****************************************************")
            print("TRAIN EPISODE", episode, "/", n_episodes)
            print("epsilon:", self.epsilon)

            # Reset the environment
            state = self.env.reset()
            done = False
            total_rewards = 0

            action = self.choose_action(state)
            while True:
                # Apply action and observe the new_state and reward
                new_state, reward, done, info = self.env.step(action)
                next_action = self.choose_action(new_state)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                # qtable[new_state,:] : all the actions we can take from new state
                self.update_qtable(state, action, new_state, next_action, reward)

                total_rewards += reward

                # Update state & action
                state = new_state
                action = next_action

                # If done, finish episode
                if done == True: 
                    break

            print()
            # Reduce epsilon (because we need less and less exploration)
            # epsilon = parameters['min_epsilon'] + (parameters['max_epsilon'] - parameters['min_epsilon'])*np.exp(-parameters['decay_rate']*episode) 
            self.epsilon = self.epsilon * self.epsilon_decay
            if self.mode == SOFTMAX:
                self.tau = self.tau * self.tau_inc
            self.rewards.append(total_rewards)

    def run(self, n_episodes):
        self.env.reset()

        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            print("****************************************************")
            print("RUN EPISODE", episode, "/", n_episodes)

            while True:
                # Take the action (index) that have the maximum expected future reward given that state
                action = np.argmax(self.qtable[state,:])
                # action = self.choose_action(state, )
                print(self.env.get_action_meanings()[action], end=" ", flush=True)

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

    def print_commands(self):
        print("     Commands: agent.train(X_episodes) | agent.print_infos() | agent.run(X_episodes) | agent.reset()")

    # Utils
    def softmax(self, tau, q):
        # https://en.wikipedia.org/wiki/Softmax_function#Reinforcement_learning
        assert tau >= 0.0
        q_tilde = q - np.max(q)
        factors = np.exp(tau * q_tilde)
        return factors / np.sum(factors)