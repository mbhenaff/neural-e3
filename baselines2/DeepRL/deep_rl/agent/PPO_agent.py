#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

        if config.rnd == 1:
            n_hidden = 500
            self.rnd_network = nn.Sequential(nn.Linear(config.state_dim, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, n_hidden),
                                             nn.ReLU(),
                                             nn.Linear(n_hidden, n_hidden)).cuda()

            self.rnd_pred_network = nn.Sequential(nn.Linear(config.state_dim, n_hidden),
                                                  nn.ReLU(),
                                                  nn.Linear(n_hidden, n_hidden),
                                                  nn.ReLU(),
                                                  nn.Linear(n_hidden, n_hidden)).cuda()
            self.rnd_optimizer = config.optimizer_fn(self.rnd_pred_network.parameters())
        

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))

            if config.rnd == 1:
                self.rnd_optimizer.zero_grad()
                s = torch.from_numpy(config.state_normalizer(states)).cuda().float()
                rnd_target = self.rnd_network(s).detach()
                rnd_pred = self.rnd_pred_network(s)
                rnd_loss = F.mse_loss(rnd_pred, rnd_target, reduction='none').mean(1)
                (rnd_loss.mean()).backward()
                self.rnd_optimizer.step()
                rewards += config.rnd_bonus*rnd_loss.detach().cpu().numpy()
            
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]


                if config.rnd == 1:
                    self.rnd_optimizer.zero_grad()
                    s = config.state_normalizer(sampled_states).cuda().float()
                    rnd_target = self.rnd_network(s).detach()
                    rnd_pred = self.rnd_pred_network(s)
                    rnd_loss = F.mse_loss(rnd_pred, rnd_target, reduction='none').mean(1)
                    (rnd_loss.mean()).backward()
                    self.rnd_optimizer.step()
                

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()
