import sys
import gym
import torch
import random
from replay_memory import ReplayMemory
from qNetwork import QNetwork
from convolutionalQNetwork import ConvQNetwork
from gym import wrappers, logger
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optimizer
import copy
from wrappers import GreyScale_Resize, FrameSkippingMaxing, StackFrames


class Agent:
    def __init__(self, action_space, observation_space, learning_rate, alpha=0.01, from_file=False):
        self.alpha = alpha
        self.action_space = action_space
        self.observation_space = observation_space

        self.q_network = QNetwork(observation_space.shape[0], action_space.n, 0, 64, 64)
        # self.target_network = copy.deepcopy(self.q_network)
        self.target_network = QNetwork(observation_space.shape[0], action_space.n, 0, 64, 64)

        if from_file:
            print("Reading weights from file")
            self.q_network.load_state_dict(torch.load("saved_params/cart_pole.pt"))

        self.optimizer = optimizer.Adam(self.q_network.parameters(), lr=learning_rate)
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.q_network.cuda()
            self.target_network.cuda()
        else:
            self.device = "cpu"

    def act(self, observation, epsilon_):
        # observation = np.swapaxes(observation, 0, 2)
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_actions = self.q_network(observation)
        self.q_network.train()
        if random.random() > epsilon_:
            return np.argmax(q_actions.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_space.n))

    def learn(self, experiences, horizon):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        criterion = torch.nn.MSELoss()
        self.q_network.train()
        self.target_network.eval()
        # shape of output from the model (batch_size,action_dim) = (64,4)
        self.optimizer.zero_grad()

        predicted_targets = self.q_network(states).gather(1, actions)
        # predicted_targets = self.q_network(states)

        with torch.no_grad():
            labels_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            # labels_next = self.q_network(next_states)
        # print(labels_next)
        labels = rewards + (horizon * labels_next * (1 - dones))
        # print(labels)

        loss = criterion(predicted_targets, labels).to(self.device)
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.update_target()

    def update_target(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(self.target_network.parameters(),
                                             self.q_network.parameters()):
            target_param.data.copy_(self.alpha * local_param.data + (1 - self.alpha) * target_param.data)

    def save_param(self):
        torch.save(self.q_network.state_dict(), "saved_params/cart_pole.pt")
