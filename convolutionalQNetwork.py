import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvQNetwork(nn.Module):
    def __init__(self, num_actions):
        super(ConvQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        # self.maxp1 = nn.MaxPool2d(32, 32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # self.maxp2 = nn.MaxPool2d(16, 16)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # self.maxp3 = nn.MaxPool2d(8, 8)

        self.fc1 = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

    def forward(self, x):
        # x = F.relu(self.maxp1(self.conv1(x)))
        # x = F.relu(self.maxp2(self.conv2(x)))
        # x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Dueling_DQN(nn.Module):
    def __init__(self, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x
