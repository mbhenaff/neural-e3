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
        ## NOTE: all environments have max_reward 1
        self.max_reward = 1
        
        if 'horizon' in params.keys():
            self.horizon=params['horizon']
        
        self.num_actions = actions
        self.traj = []
        self.replay_buffer = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'levels': []}
        self.current_level = 0
        self.started_training = False
        self.action_buffer = []
        self.mode = 'explore'

        print("[NeuraE3] Initialized")

    def select_action(self, x, level):
        if self.started_training:
            if len(self.action_buffer) == 0:
                self.action_buffer = mcts.run(self.model, torch.tensor(x),
                                              max_horizon=self.horizon,
                                              playouts=self.params['n_playouts'],
                                              n_ensemble=self.params['n_ensemble'],
                                              n_samples=self.params['n_samples'],
                                              mode=self.mode,
                                              orig_level = level)
            action = self.action_buffer[0]
            self.action_buffer = []
#            self.action_buffer = self.action_buffer[1:]
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
            if False:
                pred_next_states, pred_next_obs, pred_reward, _ = self.model(inputs_)
                s_labels = target_states_[:, :3].max(1)[1]
                x_labels = target_states_[:, 3:]
                loss_s = F.nll_loss(pred_next_states, s_labels)
                loss_x = F.binary_cross_entropy(pred_next_obs, x_labels)
                loss_r = F.mse_loss(pred_reward, target_rewards_)
                (loss_s + loss_x + loss_r).backward()
            else:
                pred_next_obs, pred_reward, _ = self.model(inputs_)
                loss_x = F.binary_cross_entropy(pred_next_obs, target_states_)
                loss_r = F.mse_loss(pred_reward, target_rewards_)
                (loss_x + loss_r).backward()
                
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()
#        print(f'state loss: {loss_s.item():.4f}, obs loss: {loss_x.item():.4f}')

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
            
        if len(self.replay_buffer['states']) > self.params['batch_size']:
            self.train_model(n_updates = self.params['n_model_updates'])
            self.started_training = True
        self.traj = []
        print(self.mode)

    def state_to_str(self,x):
        return("".join([str(z) for z in x]))
