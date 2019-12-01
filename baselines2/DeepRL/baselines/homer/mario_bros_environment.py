import torch, pdb, numpy
import skimage
import gym

from torch.autograd import Variable
from environments.abstract_environment import AbstractEnvironment


class MarioBrosEnvironment(AbstractEnvironment):

    def __init__(self, config):

        self.config = config
        if config.env == 'mario':
            from gym_super_mario_bros.actions import RIGHT_ONLY
            from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
            import gym_super_mario_bros
            env = gym_super_mario_bros.make('SuperMarioBros-v0')
            self.env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
        elif config.env == 'montezuma':
            self.env = gym.make('MontezumaRevengeDeterministic-v0')
        self.state_buffer = []
        self.reward_buffer = []
        self.counter = 0

    def process_image(self, s, subsample=4):
        if self.config.env == 'mario':
            s = skimage.color.rgb2gray(s)
            s = skimage.transform.resize(s, (s.shape[0] / subsample, s.shape[1] / subsample), anti_aliasing=True, mode='constant')
            s = torch.from_numpy(s)
        elif self.config.env == 'montezuma':
            s = s[34:34 + 160, :160]
            s = skimage.color.rgb2gray(s)
            s = skimage.transform.resize(s, (s.shape[0] / subsample, s.shape[1] / subsample), anti_aliasing=True, mode='constant')
            s = torch.from_numpy(s).float()
        return s 


    def reset(self):
        self.counter = 0
        self.state_buffer = []
        for _ in range(self.config.n_input_frames):
            state = self.process_image(self.env.reset(), self.config.image_subsample)
            self.state_buffer.append(state)

        return torch.stack(self.state_buffer)

    def step(self, action):
        total_reward = 0
        for _ in range(self.config.n_action_repeat):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        state = self.process_image(state, self.config.image_subsample)
        self.state_buffer.append(state)
        self.state_buffer = self.state_buffer[-self.config.n_input_frames:]
        return torch.stack(self.state_buffer), total_reward, done, info
