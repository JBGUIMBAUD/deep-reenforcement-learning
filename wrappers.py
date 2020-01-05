import gym
import numpy as np
import random
import cv2
from collections import deque

cv2.ocl.setUseOpenCL(False)


class GreyScale_Resize(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        """
        Put frames in greyscales and size 84*84 .
        """
        super().__init__(env)
        self._width = width
        self._height = height

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, 1),
            dtype=np.uint8,
        )
        original_space = self.observation_space
        self.observation_space = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        frame = obs
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        frame = np.expand_dims(frame, -1)
        obs = frame
        return obs


class FrameSkippingMaxing(gym.Wrapper):
    def __init__(self, env, gap=4):
        super().__init__(env)
        self._gap = 4
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        done = None
        info = {}
        for i in range(self._gap):
            obs, reward, done, info = self.env.step(action)
            if i == self._gap - 2:
                self._obs_buffer[0] = obs
            if i == self._gap - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break

        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class StackFrames(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shape[:-1] + (shape[-1] * k,)),
                                                dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self._k):
            self._frames.append(ob)
        return np.concatenate(self._frames, axis=2)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self._frames.append(ob)
        return np.concatenate(self._frames, axis=2), reward, done, info


class FireReset(gym.Wrapper):
    def __init__(self, env):
        """Fire at reset"""
        super().__init__(env)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLife(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)
