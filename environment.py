import torch, pdb, numpy, utils, random, skimage, maze, re
import gym

from torch.autograd import Variable

class EnvironmentWrapper:

    def __init__(self, config):

        self.config = config
        if config.env == 'mountaincar':
            self.env = gym.make('MountainCar-v0')
        elif config.env == 'acrobot':
            self.env = gym.make('Acrobot-v1')
        elif config.env == 'acrobotsp1':
            self.env = gym.make('Acrobot-v1')
            def _terminal(env):
                s = self.state
                return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.25)
        elif config.env == 'acrobotsp2':
            self.env = gym.make('Acrobot-v1')
            def _terminal(env):
                s = self.state
                return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.5)
        elif 'maze' in config.env:
            self.env = maze.MazeEnv(size=config.maze_size, time=100, holes=0, num_goal=1)
        elif 'montezuma-ram' in config.env:
            self.env = gym.make('MontezumaRevenge-ram-v4')
        self.state_buffer = []
        self.reward_buffer = []
        self.counter = 0
        self.smax = torch.tensor(self.env.observation_space.high)
        self.smin = torch.tensor(self.env.observation_space.low)
        
    def normalize(s):
        return (s - self.smin) / (self.smax - self.smin)

    def unnormalize(s):
        return s * (self.smax - self.smin) + self.smin
        
    def process_state(self, s, subsample=4):
        if self.config.env == 'mountaincar' or ('acrobot' in self.config.env):
            s = torch.from_numpy(s).float().squeeze()
            s = (s - self.smin) / (self.smax - self.smin)
        elif 'maze' in self.config.env:
            s = torch.from_numpy(s).permute(2, 0, 1).clone().float()
        elif 'montezuma-ram' in self.config.env:
            s_ = [s[3], s[42], s[43], s[52], s[27], s[83], s[0], s[55], s[67], s[47]]
            s_ = [float(x) for x in s_]
            s = torch.tensor(s_).float() / 255.0
        return s 


    def reset(self):
        self.counter = 0
        self.state_buffer = []
        for _ in range(self.config.n_input_frames):
            state = self.process_state(self.env.reset(), self.config.image_subsample)
            self.state_buffer.append(state)
        return torch.stack(self.state_buffer)

    def step(self, action):
        total_reward = 0
        for _ in range(self.config.n_action_repeat):
            state, reward, done, info = self.env.step(action)[:4]
            if self.config.env == 'mountaincar':
                # this environment always returns reward -1, so hardcode reward 0 once done
                position, velocity = state
                reward = 0.0 if (position >= self.env.env.goal_position) else -1.0
            elif 'acrobot' in self.config.env:
                reward = 0.0 if self.env.env._terminal() else -1.0
            total_reward += reward
            if done:
                break
        state = self.process_state(state, self.config.image_subsample)
        self.state_buffer.append(state)
        self.state_buffer = self.state_buffer[-self.config.n_input_frames:]
        return torch.stack(self.state_buffer), total_reward, done, info


def add_constants(config):
    # Add constants specific to environment
    if 'maze' in config.env:
        config.maze_size = int(re.findall(r'\d+', config.env)[0])
        config.n_input_channels = 3
        config.n_input_frames = 1
        config.n_actions = 4
        config.height = config.maze_size
        config.width = config.maze_size
        config.image_subsample = 1
        config.phi_layer_size = 25 * config.n_feature_maps
        config.n_action_repeat = 1
        config.edim = config.phi_layer_size
        config.rmin = -0.5
        config.rmax = 2
        config.max_exploration_steps = 1000
        config.input_type = 'image'
        config.phi = 'learned'
    elif config.env == 'mountaincar':
        config.n_input_channels = 2
        config.n_input_frames = 1
        config.n_actions = 3
        config.height = 1
        config.width = 1
        config.edim = 2
        config.image_subsample = 1
        config.phi_layer_size = 2
        config.n_action_repeat = 1
        config.phi='none'
        config.spherenorm = 0
        config.learn_radius = 0
        config.rmin = -1.0
        config.rmax = 0.0
        config.input_type = 'features'
    elif 'acrobot' in config.env:
        config.n_input_channels = 6
        config.n_input_frames = 1
        config.n_actions = 3
        config.height = 1
        config.width = 1
        config.edim = 6
        config.image_subsample = 1
        config.phi_layer_size = 6
        config.n_action_repeat = 1
        config.phi='none'
        config.spherenorm = 0
        config.learn_radius = 0
        config.rmin = -1.0
        config.rmax = 0.0
        config.input_type = 'features'
    elif 'montezuma-ram' in config.env:
        config.n_input_channels = 10
        config.n_input_frames = 1
        config.n_actions = 18
        config.height = 1
        config.width = 1
        config.edim = 10
        config.image_subsample = 1
        config.phi_layer_size = 10
        config.n_action_repeat = 4
        config.phi='none'
        config.spherenorm = 0
        config.learn_radius = 0
        config.rmin = 0.0
        config.rmax = 1.0
        config.input_type = 'features'
        config.max_exploration_steps = 10000
    else:
        ValueError
    return config
    
