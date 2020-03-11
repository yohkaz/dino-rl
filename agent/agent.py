import numpy as np
import random

# Action policies
EPSILON = "EPSILON"
SOFTMAX = "SOFTMAX"

class Agent:
    def __init__(self, env):
        self.env = env
        self.qtable = np.zeros((env.observation_space.high[0], env.observation_space.high[1], env.action_space.n))
        self.trained_episodes = 0

    def set_parameters(self, parameters):
        self.parameters = parameters

    def set_qtable(self, qtable):
        self.qtable = qtable

    def choose_action(self, state):
        action = None
        if self.parameters['policy'] == EPSILON:
            if random.uniform(0, 1) > self.parameters['epsilon']:
                print("(B)", end="")
                # action = np.argmax(self.qtable[state, :])
                action = np.argmax(self.qtable[state])
            else:
                print("(R)", end="")
                # random choice --> exploration
                action = self.env.action_space.sample()
        elif self.parameters['policy'] == SOFTMAX:
            # Compute probabilities for each action
            # prob_a = self.softmax(self.parameters['tau'], self.qtable[state, :])
            prob_a = self.softmax(self.parameters['tau'], self.qtable[state])
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
        rewards = []

        for episode in range(n_episodes):
            print("****************************************************")
            print("TRAIN EPISODE", episode, "/", n_episodes-1)
            if self.parameters['policy'] == EPSILON:
                print("epsilon:", self.parameters['epsilon'])
            elif self.parameters['policy'] == SOFTMAX:
                print("tau:", self.parameters['tau'])

            # Reset the environment
            state = self.env.reset()
            done = False
            total_rewards = 0

            action = 1
            while True:
                # Apply action and observe the new_state and reward
                new_state, reward, done, info = self.env.step(action)
                if new_state == state and done == False:
                    continue
                # if True:
                #     print()
                #     print(state, " // ", new_state)
                #     print()
                next_action = self.choose_action(new_state)

                # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
                self.update_qtable(state, action, new_state, next_action, reward)

                total_rewards += reward

                # Update state & action
                state = new_state
                action = next_action

                # If done, finish episode
                if done == True:
                    print()
                    print("Total rewards:", total_rewards)
                    break

            self.trained_episodes += 1
            print()

            # Reduce epsilon (because we need less and less exploration)
            if self.parameters['policy'] == EPSILON:
                self.parameters['epsilon'] = self.parameters['epsilon'] * self.parameters['epsilon_decay']
            elif self.parameters['policy'] == SOFTMAX:
                self.parameters['tau'] = self.parameters['tau'] * (1 + self.parameters['tau_inc'])
            rewards.append(total_rewards)

        return self.qtable, rewards

    def run(self, n_episodes):
        scores = []
        self.env.reset()

        for episode in range(n_episodes):
            state = self.env.reset()

            done = False
            print("****************************************************")
            print("RUN EPISODE", episode, "/", n_episodes-1)

            while True:
                # Take the action (index) that have the maximum expected future reward given that state
                # action = np.argmax(self.qtable[state,:])
                action = np.argmax(self.qtable[state])
                # action = self.choose_action(state, )
                print(self.env.get_action_meanings()[action], end=" ", flush=True)

                new_state, reward, done, info = self.env.step(action)

                if done:
                    self.env.render()
                    score = self.env.get_score()
                    print()
                    print("Score:", score)
                    scores.append(score)
                    break
                state = new_state
        return scores

    def reset(self):
        self.env.relaunch_game()

    def print_infos(self):
        print("Parameters:")
        print(self.parameters)
        print("Trained episodes:")
        print(self.trained_episodes)
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