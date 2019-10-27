import random, pdb, math, copy, numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.autograd import Variable
from experience import Rollout
from experience import ReplayMemory
import search, utils
import models

            

class Agent(nn.Module):

    def __init__(self, config):
        super(Agent, self).__init__()
        self.config = config

        # replay memory
        self.replay_memory = {'explore': {'train': ReplayMemory(config), 'test': ReplayMemory(config)},
                              'exploit': {'train': ReplayMemory(config), 'test': ReplayMemory(config)}}
                              
        # forward model
        if 'maze' in config.env:
            self.forward_model = models.ForwardModelImageEnsembleSmall(config)
        else:
            self.forward_model = models.ForwardModelEnsembleGPU(config)


    def compute_uncertainty(self, state, action, encode=False):
        bsize = state.size(0)
        phi_state = state
        next_phi_state_samples, _ = self.forward_model.forward_all(phi_state, action)
        next_phi_state_samples = next_phi_state_samples.squeeze()
        uncertainty_estimates = utils.compute_uncertainty(next_phi_state_samples)
        return uncertainty_estimates

    # perform rollout of an action sequence, returns prediction and uncertainty
    def rollout(self, state, actions):
#        state = state.view(-1, self.config.n_input_frames, self.config.n_input_channels, self.config.height, self.config.width)
#        state = self.phi_network(state.float().cuda())
        if self.config.input_type == 'features':
            state = state.view(-1, self.config.edim)
        u_list, s_list = [], []
        for a in actions:
            action = utils.one_hot(a, self.config.n_actions)
            state, _ = self.forward_model.forward_all(state, action, particles=True)
            u = torch.var(state.squeeze(), 0).sum().item()
            u_list.append(u)
            s_list.append(state.detach())
        return torch.stack(s_list), torch.tensor(u_list)

    # calculate predictions, uncertainty and error over replay buffer using current model
    def estimate_stats(self, split, goal, config):
        replay_memory = self.replay_memory[goal][split]
        print('[estimating uncertainty statistics]')
        nsamples = min(replay_memory.n_samples(), 10)
        u_list      = torch.zeros(nsamples, config.max_exploration_steps)
        mse_list    = torch.zeros(nsamples, config.max_exploration_steps)
        if self.config.input_type == 'image':
            s_pred_list = torch.zeros(nsamples, config.max_exploration_steps, config.n_ensemble, config.n_input_channels, config.width, config.height)
            s_real_list = torch.zeros(nsamples, config.max_exploration_steps, config.n_input_channels, config.width, config.height)
        else:
            s_pred_list = torch.zeros(nsamples, config.max_exploration_steps, config.n_ensemble, config.edim)
            s_real_list = torch.zeros(nsamples, config.max_exploration_steps, config.edim)

        a_list = torch.zeros(nsamples, config.max_exploration_steps)
        ep_length_list = []
        for i in range(nsamples):
            print(f'[processing replay buffer sample {nsamples-i-1}]')
            rollout = replay_memory.rollout(nsamples-i-1)
            state = rollout.states[0]
            actions = rollout.actions
            ep_length_list.append(len(rollout.states))
            s_real = torch.stack(rollout.states).squeeze()
                
            state = state.unsqueeze(0)
            state = self.phi_network(state.cuda())
            if state.dim() == 4:
                state_rep = state.repeat(config.n_ensemble, 1, 1, 1)
            else:
                state_rep = state.repeat(config.n_ensemble, 1)
            s_pred, u = self.rollout(state_rep, actions, particles=True)
            if self.config.input_type == 'image' and (config.phi == 'learned' or config.phi == 'ae'):
                decoded_imgs = []
                for j in range(s_pred.size(0)):
                    pred = self.phi_decoder(s_pred[j]).detach()
                    if config.loss == 'softmax':
                        img=torch.multinomial(torch.exp(pred.view(-1, 256)).contiguous(), num_samples=1)
                        img = img.view(self.config.n_ensemble, self.config.n_input_channels, self.config.height, self.config.width)
                        decoded_imgs.append(img.cpu())
                    else:
                        decoded_imgs.append(pred.cpu())
                s_pred = torch.stack(decoded_imgs)
                mse = torch.zeros(1)
            else:
                s_pred_targets = self.phi_network(s_real.cuda())
                mse = F.mse_loss(s_pred_targets.unsqueeze(1)[1:].expand(s_pred.size()).cuda(), s_pred, reduction='none').detach()
            mse = mse.view(mse.size(0), -1).mean(1)
                
            if u.numel() == config.max_exploration_steps:
                u_list[i].copy_(u)
                mse_list[i].copy_(mse)
                s_real_list[i].copy_(s_real[1:])
                s_pred_list[i].copy_(s_pred)
#                a_list[i][:len(actions)].copy_(torch.tensor(actions))
            elif u.numel() < config.max_exploration_steps:
                u_list[i][:u.numel()].copy_(u)
                u_list[i][u.numel():].copy_(u[-1])
                mse_list[i][:mse.numel()].copy_(mse)
                mse_list[i][mse.numel():].copy_(mse[-1])
                s_pred_list[i][:s_pred.size(0)].copy_(s_pred)
                s_real_list[i][:s_real.size(0)-1].copy_(s_real[1:])
#                a_list[i][:len(actions)].copy_(torch.tensor(actions))
                
        u_list = u_list.cpu()
        mse_list = mse_list.cpu()
        s_real_list = s_real_list.cpu()
        s_pred_list = s_pred_list.cpu()
        if self.config.u_quantile == -1:
            eps = self.config.eps
        else:
            eps = torch.sort(u_list.view(-1), descending=False)[0][round(config.u_quantile*u_list.numel())]
        self.eps = eps
        stats = {'uncertainty': u_list,
                 'mse': mse_list,
                 's_real': s_real_list,
                 's_pred': s_pred_list,
                 'actions': a_list,
                 'ep_length': ep_length_list,
                 'eps': eps}
        return stats
                

    

    # return a batch of trajectory segments of length T from the replay memory
    def get_batch_from_replay_memory(self, replay_memory, config, T, batch_size, nonzero_reward_only=False, p_last=0.5, p_nonzero=0.25):
        states, next_states, actions, rewards, returns, timesteps = [], [], [], [], [], []
        b = 0
        while b < batch_size:
            # pick trajectory. If we have nonzero rewards in the buffer, sample them with decent prob
            ok = False
            if nonzero_reward_only:
                if random.random() < p_nonzero:
                    rollout, indx = replay_memory.sample_nonzero_reward()
                    states_, next_states_, actions_, rewards_, returns_, timesteps_ = rollout.sample_sequence(T)
                    ok = True                
                else:
                    rollout = replay_memory.importance_sample(p=p_last, n = config.n_trajectories)
                    if rollout.length >= T + config.n_input_frames:
                        states_, next_states_, actions_, rewards_, returns_, timesteps_ = rollout.sample_sequence(T)
                        ok = True
                    
            elif len(replay_memory.reward_indices) > 0 and random.random() < p_nonzero:
                rollout, indx = replay_memory.sample_nonzero_reward()
                states_, next_states_, actions_, rewards_, returns_, timesteps_ = rollout.sample_sequence(T=T, t0 = indx.item() - T + 1)
                ok = True
            else:
                rollout = replay_memory.importance_sample(p=p_last, n = config.n_trajectories)
                if rollout.length >= T + config.n_input_frames:
                    states_, next_states_, actions_, rewards_, returns_, timesteps_ = rollout.sample_sequence(T)
                    ok = True
            if ok:
                states.append(states_)
                next_states.append(next_states_)
                actions.append(actions_)
                rewards.append(rewards_)
                returns.append(returns_)
                timesteps.append(timesteps_)
                b += 1

        states = torch.stack(states).cuda()
        next_states = torch.stack(next_states).cuda()
        actions = torch.stack(actions).cuda()
        rewards = torch.stack(rewards).cuda()
        returns = torch.stack(returns).cuda()
        timesteps = torch.stack(timesteps).cuda()

        # watch out for broadcasting bugs!
        if self.config.input_type == 'image':
            states = states.view(batch_size, T, self.config.n_input_channels*self.config.n_input_frames, self.config.height, self.config.width)
            next_states = next_states.view(batch_size, T, self.config.n_input_channels*self.config.n_input_frames, self.config.height, self.config.width)

        # this can be useful for debugging, to make sure the forward model is using actions
        if config.zeroact == 1:
            actions.data.zero_()
        
        return states, next_states, actions, rewards, returns, timesteps
    
    def calc_loss(self, replay_memory, config, batch_size):
        assert(replay_memory.n_samples() > 0)
        states, _, actions, rewards, _, _ = self.get_batch_from_replay_memory(replay_memory, config, config.T+1, batch_size)
        losses = {}

        s = states[:, 0]
        phi = s
        states_pred, r_pred, logprob_a = [], [], []
        
        for t in range(config.T+1):
            s_next, r_next = self.forward_model(s, actions[:, t])
            states_pred.append(s_next)
            r_pred.append(r_next)
            s = s_next
        states_pred = torch.stack(states_pred, 1)
        r_pred = torch.stack(r_pred, 1)
        states_targets = states
        states_pred = states_pred[:, :-1]
        losses['fwd_loss_s'] = F.mse_loss(states_pred, states_targets[:, 1:].contiguous().view(states_pred.size()))
        losses['fwd_loss_r'] = F.mse_loss(r_pred, rewards) * self.config.lambda_r
            

        # record performance for positive and negative rewards
        '''
        r_pred_pos = r_pred[rewards.eq(config.rmax)].mean()
        r_pred_neg = r_pred[rewards.eq(config.rmin)].mean()
        metrics = {'r_pred_pos': r_pred_pos,
                   'r_pred_neg': r_pred_neg}
        '''
        metrics = {}
        return losses, metrics
    

    def train_policy_dqn(self, split, goal, optimizer, config, n_updates = 1000, logger = None):
        losses = []
        for j in range(n_updates):
            optimizer.zero_grad()
            self.zero_grad()
            self.dqn.zero_grad()
#            states, _, actions, rewards, returns, timesteps = self.get_batch_from_replay_memory(self.replay_memory[goal][split], config, 2, config.batch_size, nonzero_reward_only=True, p_last=0.0, p_nonzero=config.dqn_pos_reward_prob)
            states, _, actions, rewards, returns, timesteps = self.get_batch_from_replay_memory(self.replay_memory[goal][split], config, 2, config.batch_size, nonzero_reward_only=False, p_last=0.0, p_nonzero=config.dqn_pos_reward_prob)
            states, next_states = states[:, 0].squeeze(), states[:, 1].squeeze()
            actions = actions[:, 0].max(1)[1]
            rewards = rewards[:, 1] # TODO
            terminals = rewards + 1.0 # TODO hardcoded
            rewards = rewards * config.rscale
            qvals, loss = self.dqn(states, actions, next_states, rewards, terminals)
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), config.dqn_grad_clip)
            optimizer.step()
            if j % config.dqn_update_target_freq == 0:
                self.dqn.sync_networks()
            if logger is not None and j > 1000 and j % 1000 == 0:
                logger.log(f'DQN training step {j}, loss: {numpy.mean(losses[-1000:]):.5f}')
        return numpy.mean(losses)

    def train_policy_dqn_with_model(self, split, goal, optimizer, config, forward_model, n_updates = 100, reward_scale=1.0, grad_clip=10.0, nsteps=3):
        total_loss, total_r = 0, 0
        for j in range(n_updates):
            # regular updates
            optimizer.zero_grad()
            self.zero_grad()
            self.dqn.zero_grad()
            states, _, actions, rewards, returns, timesteps = self.get_batch_from_replay_memory(self.replay_memory[goal][split], config, 2, config.batch_size, nonzero_reward_only=True, p_last=0.0, p_nonzero=0.1)
            states, next_states = states[:, 0].squeeze(), states[:, 1].squeeze()
            actions = actions[:, 0].max(1)[1]
            rewards = rewards[:, 1] # TODO
            terminals = rewards + 1.0 # TODO hardcoded
            rewards = rewards * reward_scale
            qvals, loss = self.dqn(states, actions, next_states, rewards, terminals)
            loss.backward()
            total_loss += loss.item()
            total_r += rewards.sum().item()
            torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), grad_clip)
            optimizer.step()
            # model updates
            optimizer.zero_grad()
            self.zero_grad()
            self.dqn.zero_grad()
            states, _, actions, rewards, returns, timesteps = self.get_batch_from_replay_memory(self.replay_memory[goal][split], config, nsteps+1, config.batch_size, nonzero_reward_only=True, p_last=0.0, p_nonzero=0.1)
            states, next_states = states[:, 0].squeeze(), states[:, 1].squeeze()
            actions = actions[:, 0].max(1)[1]
            rewards = rewards[:, 1] # TODO
            terminals = rewards + 1.0 # TODO hardcoded
            assert(terminals.sum() == 0)
            rewards = rewards * reward_scale
            qvals, loss = self.dqn.forward_nstep(states, actions, next_states, rewards, terminals, forward_model, nsteps=3)
            loss.backward()
            total_loss += loss.item()
            total_r += rewards.sum().item()
            torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), grad_clip)
            optimizer.step()
        return total_loss / n_updates, total_r / n_updates
    
            
    def train_model(self, split, goal, optimizer, config, tensorboard, update=True):
        replay_memory = self.replay_memory[goal][split]
        loss_terms = {}
        metrics_terms = {}
        for i in range(config.n_model_updates):
            # Update the policy and version space
            optimizer.zero_grad()
            self.zero_grad()
            #batch_size = config.batch_size if update else 100
            batch_size = config.batch_size
            losses, metrics = self.calc_loss(replay_memory, config, batch_size)
            loss = 0
            for k, v in losses.items():
                if v is not None:
                    loss = loss + v
                    if i == 0:
                        loss_terms[k] = v.item()
                    else:
                        loss_terms[k] += v.item()

            for k, v in metrics.items():
                if v is not None:
                    if i == 0:
                        metrics_terms[k] = v.item()
                    else:
                        metrics_terms[k] += v.item()
                        
            if update:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
                optimizer.step()

        log_string = f'{split}/{goal} | trajectories: {replay_memory.n_samples()}'
        for lk, lv in loss_terms.items():
            lv = lv / config.n_model_updates
            log_string += f' | {lk}: {lv:.5f}'

        for lk, lv in metrics_terms.items():
            lv = lv / config.n_model_updates
            log_string += f' | {lk}: {lv:.5f}'
                            
        return loss_terms, log_string


    def act(self, env, split, config, policy='random', goal='explore', stats=None, n_episodes=None):
        replay_memory = self.replay_memory[goal][split]

        if n_episodes is None:
            n_episodes = self.config.n_trajectories

        # method we will use for choosing actions
        if policy == 'random':
            self.search_method = search.UniformExploration(config)
        elif policy == 'particle2':
            self.search_method = search.ParticleSearch2(config)
        else:
            self.search_method = None

        ep_reward, ep_length = [], []
        for episode in range(n_episodes):
            done = False
            state = env.reset()
            step = 0
            rollout = Rollout(config)
            while not done and step < config.max_exploration_steps:                    
                if self.search_method is not None:
                    action, _, _, _, _ = self.search_method.search(state, self, goal=goal, eps=config.eps)
                    if 'montezuma' in config.env:
                        if len(action) > config.T: action = action[:config.T]
                        action.append(random.randint(0, config.n_actions - 1))
                elif policy == 'dqn':
                    qvals, _ = self.dqn(state.cuda())
                    action = qvals.squeeze().max(0)[1]
                    action = [action.item()]
                for a in action:
                    next_state, reward, done, info = env.step(a)
                    step += 1
                    rollout.append(state, a, reward)
                    if done or step >= config.max_exploration_steps:
                        rollout.append(next_state, None, None)
                        break
                    else:
                        state = next_state

            # Save the data
            replay_memory.add(rollout)
            ep_reward.append(torch.tensor(rollout.rewards).sum().item())
            ep_length.append(step)
        ep_reward = numpy.mean(ep_reward)
        ep_length = numpy.mean(ep_length)
#        summary = {'reward': torch.tensor(ep_reward).mean(),
#                   'stats': stats}
        return ep_reward, ep_length
