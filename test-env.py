import argparse
import sys

import gym
from gym import wrappers, logger
# from atari_wrapper_sources import NoopResetEnv, MaxAndSkipEnv, wrap_deepmind
from wrappers import GreyScale_Resize, FrameSkippingMaxing, StackFrames


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        # print(observation)
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='BreakoutNoFrameskip-v4', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)
    env = GreyScale_Resize(env)
    env = FrameSkippingMaxing(env)
    # env = StackFrames(env)

    # env = NoopResetEnv(env, noop_max=30)
    # env = MaxAndSkipEnv(env, skip=4)
    # env = wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=False, scale=False)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    # env = make_wrap_atari(args.env_id)
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    # for i in range(len(env.unwrapped.get_action_meanings())):
    #     print(env.unwrapped.get_action_meanings()[i])

    for i in range(episode_count):
        ob = env.reset()
        # print(ob.shape)
        j = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            # print(ob.shape)
            if done:
                break
            j = j + reward
        print(j)
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
