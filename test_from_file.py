import argparse
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
from wrappers import GreyScale_Resize, FrameSkippingMaxing, StackFrames, FireReset, EpisodicLife, ClipRewardEnv, \
    NoopReset
from breakout_agent import Agent as BAgent
from cartPole_agent import Agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='BreakoutNoFrameskip-v4', help='Select the environment to run. '
                                                                                    'Default beeing '
                                                                                    'BreakoutNoFrameskip-v4')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    if args.env_id == "CartPole-v1":
        env = gym.make("CartPole-v1")
        agent = Agent(env.action_space, env.observation_space, 0, 0, True)
        # model = QNetwork(env.observation_space.shape[0], env.action_space.n, 0, 64, 64)
        # model.load_state_dict(torch.load("saved_params/cart_pole.pt"))
    else:
        env = gym.make("BreakoutNoFrameskip-v4")
        # Research paper wrappers
        env = GreyScale_Resize(env)
        env = FrameSkippingMaxing(env)
        env = StackFrames(env)
        # Optimisation wrappers
        env = NoopReset(env)
        env = EpisodicLife(env)
        env = FireReset(env)
        # env = ClipRewardEnv(env)
        agent = BAgent(env.action_space, env.observation_space, 0, 0, True)
        # model = ConvQNetwork(2)
        # model.load_state_dict(torch.load("saved_params/breakout.pt"))

    env.seed(0)

    reward = 0
    learn = 0

    for i in range(10):
        state = env.reset()
        sum_reward = 0
        while True:
            action = agent.act(state, epsilon_=0)
            next_state, reward, done, _ = env.step(action)
            sum_reward = sum_reward + reward
            state = next_state
            env.render()
            if done:
                break
