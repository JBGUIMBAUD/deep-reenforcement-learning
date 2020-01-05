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
from wrappers import GreyScale_Resize, FrameSkippingMaxing, StackFrames, FireReset, EpisodicLife, ClipRewardEnv
from breakout_agent import Agent as BAgent
from cartPole_agent import Agent


def main(environment):
    BUFFER_SIZE = 50000
    BATCH_SIZE = 64  # mini-batch size
    GAMMA = 0.99  # discount factor
    ALPHA = 0.01  # for soft update of target parameters
    LEARNING_RATE = 5e-4  # learning rate
    UPDATE_EVERY = 2  # how often to update the network

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    if environment == "CartPole-v1":
        env = gym.make("CartPole-v1")
        outdir = 'logs/cartpole-agent-results'
        agent = Agent(env.action_space, env.observation_space, LEARNING_RATE, ALPHA)
    else:
        env = gym.make("BreakoutNoFrameskip-v4")
        # Research paper wrappers
        env = GreyScale_Resize(env)
        env = FrameSkippingMaxing(env)
        env = StackFrames(env)
        # Optimisation wrappers
        env = EpisodicLife(env)
        env = FireReset(env)
        env = ClipRewardEnv(env)
        outdir = 'logs/breakout-agent-results'
        agent = BAgent(env.action_space, env.observation_space, LEARNING_RATE, ALPHA)

    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    episode_count = 10000
    score = [0]
    buffer = ReplayMemory(BUFFER_SIZE)
    learn = 0
    epsilon = 1
    epsilon_decay = 0.998
    epsilon_min = 0.005
    for i in range(episode_count):
        if i % 100 == 0:
            print("episode: ", i)
        state = env.reset()
        sum_reward = 0
        # for _ in range(MAX_EPISODE_LEN):
        while True:
            action = agent.act(state, epsilon_=epsilon)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            # Store experience replay
            buffer.push(state=state, action=action, next_state=next_state, reward=reward, end=done)
            sum_reward = sum_reward + reward
            state = next_state

            # Learn
            learn = (learn + 1) % UPDATE_EVERY
            if len(buffer) >= BATCH_SIZE and learn == 0:
                sample = buffer.sample(BATCH_SIZE)
                agent.learn(sample, GAMMA)

            if done:
                score = score + [sum_reward]
                break
            if epsilon > epsilon_min:
                epsilon = epsilon * epsilon_decay
            # env.render()

    # Close the env and write monitor result info to disk
    env.close()
    plt.plot(score)
    plt.ylabel('Rewards')
    plt.show()

    # Save params into a file
    agent.save_param()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='BreakoutNoFrameskip-v4', help='Select the environment to run')
    args = parser.parse_args()

    main(args.env_id)
