import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb, random
import models, mcts

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'level'))


        
        

class NeuralE3(object):
    def get_name(self):
        return "neural-e3"

    def __init__(self,actions,params={}):
        self.params = params
        self.model = models.ForwardModel(params).cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])
        
        if 'horizon' in params.keys():
            self.horizon=params['horizon']
        
        self.num_actions = actions
        self.traj = []
        self.replay_buffer = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'levels': [], 'next_levels': []}
        self.current_level = 0
        self.started_training = False
        self.action_buffer = []
        self.mode = 'explore'
        self.ucb_c = params['ucb_c']

        self.dqn = models.DQN(params).cuda()
        self.dqn_optimizer = torch.optim.Adam(self.dqn.parameters(), lr=0.0001)
        

        print("[NeuraE3] Initialized")

    def select_action(self, x, level):
        if self.started_training and self.mode != 'exploit-dqn':
            if len(self.action_buffer) == 0:
                self.action_buffer = mcts.run(self.model, torch.tensor(x),
                                              max_horizon=self.horizon,
                                              playouts=self.params['n_playouts'],
                                              n_ensemble=self.params['n_ensemble'],
                                              n_samples=self.params['n_samples'],
                                              mode=self.mode,
                                              orig_level = level,
                                              ucb_c = self.ucb_c)
            action = self.action_buffer[0]
            self.action_buffer = []
        elif self.mode == 'exploit-dqn':
            lvl = torch.tensor([int(i == level) for i in range(self.horizon)])
            inp = torch.cat((torch.from_numpy(x).float(), lvl.float())).cuda()
            qvals, loss = self.dqn(inp)
            action = torch.max(qvals, 0)[1].item()
        else:
            action = random.randint(0, self.num_actions-1)
        return (action)

    def get_value(self,x):
        s = self.state_to_str(x)
        if s not in self.Qs.keys():
            ## This should only happen at the end of the episode so put 0 here. 
            self.Qs[s] = [0 for a in range(self.num_actions)]
            self.Ns[s] = [0 for a in range(self.num_actions)]
        Qvals = self.Qs[s]
        return(np.max(Qvals))

    def save_transition(self, state, action, reward, next_state, level):
        self.traj.append(Transition(state, action, next_state, reward, level))

        
    def train_model(self, batch_size=100, n_updates = 100):
        states  = torch.tensor(np.stack(self.replay_buffer['states'])).float()
        actions = torch.tensor(np.stack(self.replay_buffer['actions'])).float()
        levels  = torch.tensor(np.stack(self.replay_buffer['levels'])).float()
        inputs = torch.cat((states, actions, levels), 1)
        target_states = torch.tensor(np.stack(self.replay_buffer['next_states'])).float()
        target_rewards = torch.tensor(np.stack(self.replay_buffer['rewards'])).float()
        n_samples = inputs.size(0)
        batch_size = min(n_samples, batch_size)

        for i in range(n_updates):
            indx = torch.randperm(n_samples)[:batch_size]
            inputs_ = inputs[indx].cuda()
            target_states_ = target_states[indx].cuda()
            target_rewards_ = target_rewards[indx].cuda()
            self.optimizer.zero_grad()
            pred_next_obs, pred_reward, _ = self.model(inputs_)
            loss_x = F.binary_cross_entropy(pred_next_obs, target_states_)
            loss_r = F.mse_loss(pred_reward, target_rewards_)
            (loss_x + loss_r).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()
#        print(f'state loss: {loss_s.item():.4f}, obs loss: {loss_x.item():.4f}')



    def train_dqn(self, batch_size=100, n_updates = 200000):
        print('[training DQN]')
        states  = torch.tensor(np.stack(self.replay_buffer['states'])).float()
        next_states  = torch.tensor(np.stack(self.replay_buffer['next_states'])).float()
        actions = torch.tensor(np.stack(self.replay_buffer['actions'])).float()
        levels  = torch.tensor(np.stack(self.replay_buffer['levels'])).float()
        levels_int = torch.max(levels, 1)[1]
        next_levels = torch.from_numpy(np.stack([np.array([int(i == h + 1) for i in range(self.horizon)]) for h in levels_int])).float()
        states_levels = torch.cat((states, levels), 1)
        next_states_levels = torch.cat((next_states, next_levels), 1)
        rewards = torch.tensor(np.stack(self.replay_buffer['rewards'])).float()
        n_samples = states.size(0)
        batch_size = min(n_samples, batch_size)

        losses = []
        for i in range(n_updates):
            indx = torch.randperm(n_samples)[:batch_size]
            states_ = states_levels[indx].cuda()
            next_states_ = next_states_levels[indx].cuda()
            actions_ = torch.max(actions[indx], 1)[1]
            rewards_ = rewards[indx].cuda()
            levels_ = levels_int[indx].cuda()
            terminals_ = torch.tensor([int(x) for x in (levels_ == self.horizon-1)]).cuda().float()
            self.dqn_optimizer.zero_grad()
            qvals, loss = self.dqn(states_, actions_, next_states_, rewards_, terminals_)
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 10)
            self.dqn_optimizer.step()
            if not i % 1000:
                self.dqn.sync_networks()
            if i > 1000 and not i % 1000:
                print(f'DQN training step {i}, loss: {np.mean(losses[-1000:]):.5f}')
            
                
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()


    def finish_episode(self):
        for transition in self.traj:
            x = transition.state
            a = transition.action
            r = transition.reward
            xp = transition.next_state
            h = transition.level
            self.replay_buffer['states'].append(x)
            self.replay_buffer['actions'].append(np.array([int(i == a) for i in range(self.num_actions)]))
            self.replay_buffer['next_states'].append(xp)
            self.replay_buffer['rewards'].append(r)
            self.replay_buffer['levels'].append(np.array([int(i == h) for i in range(self.horizon)]))
            self.replay_buffer['next_levels'].append(np.array([int(i == h + 1) for i in range(self.horizon)]))
            
        if len(self.replay_buffer['states']) > self.params['batch_size']:
            self.train_model(n_updates = self.params['n_model_updates'])
            self.started_training = True
        self.traj = []
        print(self.mode)

    def state_to_str(self,x):
        return("".join([str(z) for z in x]))
