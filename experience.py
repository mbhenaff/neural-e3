import random, pdb, utils
import torch


class ReplayMemory:
    def __init__(self, config):
        self.rollouts = []
        self.reward_indices = []
        self.config = config

    def n_samples(self):
        return len(self.rollouts)

    def rollout(self, i):
        return self.rollouts[i]

    def importance_sample(self, p=0.5, n=5):
        if self.n_samples() <= n:
            return random.choice(self.rollouts)
        else:
            if random.random() > p:
                k = random.randint(0, self.n_samples() - n - 1)
            else:
                # sample from recent trajectories
                k = random.randint(self.n_samples() - n, self.n_samples() - 1)
            return self.rollouts[k]

    def sample_nonzero_reward(self):
        ep = random.choice(self.reward_indices)
        return self.rollouts[ep['rollout']], ep['indx']

        
    def random_rollout(self):
        return random.choice(self.rollouts)
    
    def add(self, rollout):
        self.rollouts.append(rollout)
        rewards = torch.tensor(rollout.rewards)
        # TODO hardcoded for single reward
        high_rewards = rewards.eq(self.config.rmax)
        if high_rewards.any():
            if torch.argmax(high_rewards).item() > self.config.T:
                self.reward_indices.append({'rollout': len(self.rollouts) - 1, 'indx': rewards.eq(self.config.rmax).nonzero()})

    


class Rollout:
    def __init__(self, config):
        self.config = config
        self.states = []
        self.actions = []
        self.rewards = []
        self.length = 0
        self.errors = []
        self.uncertainty = []

    def append(self, state, action=None, reward=None, error = None, u = None):
        self.states.append(state[-1])
        if action is not None:
            self.actions.append(action)
        if reward is not None:
            self.rewards.append(reward)
        if error is not None:
            self.errors.append(error)
        if u is not None:
            self.uncertainty.append(u)
        if action is not None:
            self.length += 1


    def sample_sequence(self, T=None, t0=None, gamma=0.99):
        if T is None:
            T = self.length - self.config.n_input_frames
        states, actions, next_states, rewards, returns = [], [], [], [], []
        if t0 is None:
            t0 = random.randint(self.config.n_input_frames, self.length - T)
        for t in range(t0, t0+T):
            states.append(torch.stack(self.states[t+1-self.config.n_input_frames:t+1]))
            next_states.append(torch.stack(self.states[t+2-self.config.n_input_frames:t+2]))
            actions.append(utils.one_hot(self.actions[t], self.config.n_actions, unsqueeze=False))
            rewards.append(self.rewards[t])
            decay = torch.tensor([gamma**i for i in range(0, len(self.rewards[t:]))])
            returns.append((decay*torch.tensor(self.rewards[t:])).sum())

        try:
            states = torch.stack(states)
        except:
            pdb.set_trace()
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.tensor(rewards)
        returns = torch.tensor(returns)
        time = torch.arange(t0, t0+T).float() / self.config.max_exploration_steps
        return states, next_states, actions, rewards, returns, time
