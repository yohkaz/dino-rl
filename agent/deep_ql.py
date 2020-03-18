import numpy as np
import random
from agent.agent import Agent, EPSILON, SOFTMAX

from collections import namedtuple
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory(object):
    # source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class NeuralNetwork(nn.Module):
    def __init__(self, n_actions, hidden_size=512):
        super(NeuralNetwork, self).__init__()
        # https://github.com/e3oroush/dino_run_rl_pytorch/blob/master/models.py
        class FlattenTorch(nn.Module):
            def forward(self,x):
                return x.view(x.shape[0], -1)

        image_channels = 4
        self.frame_encoder = nn.Sequential(nn.Conv2d(image_channels, 32, 8, stride=4, padding=2),
                                           nn.MaxPool2d(2),
                                           nn.ReLU(),
                                           nn.Conv2d(32,64,4,stride=2,padding=1),
                                           nn.MaxPool2d(2),
                                           nn.ReLU(),
                                           nn.Conv2d(64,64,3,stride=1,padding=1),
                                           nn.MaxPool2d(2),
                                           nn.ReLU(),
                                           FlattenTorch())
        self.qtable_estimator = nn.Sequential(nn.Linear(64, hidden_size),
                                              nn.ReLU(),
                                              nn.Linear(hidden_size, n_actions))

    def forward(self, frame):
        if next(self.parameters()).is_cuda:
            frame = frame.cuda()

        features = self.frame_encoder(frame)        # [batch_size,64]
        qvalues = self.qtable_estimator(features)   # [batch_size,nb_actions]
        return qvalues

class DeepQLAgent(Agent):
    def __init__(self, env, hidden_size=512):
        super().__init__(env)
        print("DeepQLAgent:")
        super().print_commands()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define the qtable
        self.qtable = NeuralNetwork(env.action_space.n).to(self.device)
        # Define the memory
        self.memory = ReplayMemory(50000)
        # Define Adam optimizer
        self.optimizer = optim.Adam(self.qtable.parameters(), lr=1e-5)
        # Initialize mean squared error loss
        self.criterion = nn.MSELoss()
        self.mean_losses = []

    def state_to_tensor(self, state):
        # Convert state to tensor
        state = torch.from_numpy(state)
        # Convert state to input format of NN: (n_batches, in_channels, height, width)
        state = state.view(-1, 1, state.shape[0], state.shape[1])
        return state

    def get_qtable(self, state, action=None):
        if action == None:
            return self.qtable(state).cpu().detach().numpy()[0]
        else:
            return self.qtable(state).cpu().detach().numpy()[0][action]

    def update_qtable(self, state, action, next_state, next_action, reward):
        # Sample a batch
        BATCH_SIZE = 32
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Unpack the batch
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = np.array(batch.done)

        # update Q-values for targets non-terminal states
        targets = torch.zeros(BATCH_SIZE).to(self.device)
        targets[~done_batch] = self.qtable(next_state_batch)[~done_batch].max(1)[0].detach()
        targets = (self.parameters['gamma'] * targets) + reward_batch
        # Debug
        # tmp = self.qtable(next_state_batch)[~done_batch]
        # print("DEBUG:", done_batch[0], action_batch[0], tmp[0], reward_batch[0], targets[0])
        # print("DEBUG:", done_batch[31], reward_batch[31], targets[31])

        # extract Q-value
        q_value = self.qtable(state_batch)
        # q_value = q_value[torch.arange(state_batch.shape[0]), action_batch]
        q_value = q_value.gather(1, action_batch)
        # q_value = q_value.detach()

        self.qtable.train()
        loss = self.criterion(q_value, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

    def train(self, n_episodes):
        for episode in range(n_episodes):
            self.print_episode(episode, n_episodes, True)
            # Reset the environment
            state = self.state_to_tensor(self.env.reset())
            state = torch.cat((state, state, state, state), 1) #stacking 4 images
            done = False
            total_rewards = 0
            self.losses = []
            action = 1

            while True:
                self.qtable.eval()
                # Apply action and observe the new_state and reward
                new_state, reward, done, info = self.env.step(action)
                total_rewards += reward

                # Stacking 4 images
                new_state = torch.cat((state[:, 1:, :, :], self.state_to_tensor(new_state)), 1)
                reward = torch.tensor([float(reward)])
                # action = torch.tensor([action])
                action = torch.tensor([[action]])
                # Save the transition in memory
                self.memory.push(state, action, new_state, reward, done)

                # Update qtable (neural network)
                self.update_qtable(state, action, new_state, None, reward)

                # Update state & action
                state = new_state
                action = self.choose_action(state)

                # If done, finish episode
                if done == True:
                    print()
                    print("Total rewards:", total_rewards)
                    mean_loss = np.mean(self.losses)
                    print("Mean Loss:", mean_loss)
                    break

            self.update_training_parameters()
            self.rewards.append(total_rewards)
            self.mean_losses.append(mean_loss)

        return self.qtable, self.rewards, self.mean_losses

    def run(self, n_episodes):
        scores = []

        for episode in range(n_episodes):
            state = self.state_to_tensor(self.env.reset())
            state = torch.cat((state, state, state, state), 1) #stacking 4 images
            done = False
            self.print_episode(episode, n_episodes, False)
            while True:
                # Take the action (index) that have the maximum expected future reward given that state
                action = np.argmax(self.get_qtable(state))
                print(self.env.get_action_meanings()[action], end=" ", flush=True)

                new_state, reward, done, info = self.env.step(action)
                new_state = torch.cat((state[:, 1:, :, :], self.state_to_tensor(new_state)), 1)

                if done:
                    self.env.render()
                    score = self.env.get_score()
                    print()
                    print("Score:", score)
                    scores.append(score)
                    break

                state = new_state

        return scores

    def print_infos(self):
        super().print_infos()
        print("Mean Losses:")
        print(self.mean_losses)

        # Loss plot
        import matplotlib.pyplot as plt
        plt.plot(range(len(self.mean_losses)), self.mean_losses, '-')
        plt.ylabel("loss")
        plt.xlabel("episodes")
        plt.show()

