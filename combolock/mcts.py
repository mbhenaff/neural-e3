import os, pdb, torch
import sys
import random
import itertools
from time import time
from copy import copy
from math import sqrt, log

def ucb(node, c=1.0):
    return node.value / node.visits + c*sqrt(log(node.parent.visits)/node.visits)

torch.set_default_tensor_type(torch.FloatTensor)

def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return range(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError


class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0



def run(model,
        start_state,
        mode='explore',
        n_actions=4,
        max_horizon=10,
        n_ensemble=8,
        n_samples=20,
        playouts=1000,
        orig_level=0):
    
    start_state = start_state.float()
    best_rewards = []
    start_time = time()
    root = Node(None, None)

    best_actions = []
    best_reward = float("-inf")
    start_state = start_state.unsqueeze(0).repeat(n_ensemble*n_samples, 1).cuda()

    for k in range(playouts):
        state = start_state.clone()
        samples = start_state.clone()
        sum_reward = 0
        node = root
        terminal = False
        actions = []
        # selection
        while node.children:
            if node.explored_children < len(node.children):
                child = node.children[node.explored_children]
                node.explored_children += 1
                node = child
            else:
                node = max(node.children, key=ucb)
            action = torch.tensor([i == node.action for i in range(n_actions)]).float()
            action = action.unsqueeze(0).repeat(n_ensemble*n_samples, 1).cuda()
            horizon = torch.tensor([i == (orig_level + len(actions)) for i in range(max_horizon)]).float().unsqueeze(0).repeat(n_ensemble*n_samples, 1).cuda()
            state_action = torch.cat((samples, action, horizon), 1).cuda()
#            s_probs, x_probs, reward, samples = model(state_action, n_samples=1)
            x_probs, reward, samples = model(state_action, n_samples=1)
            
            if mode == 'explore':
                probs = samples.view(n_ensemble, n_samples, -1).mean(1)
                reward = reward.view(n_ensemble, n_samples).mean(1).unsqueeze(1)
                probs = torch.cat((probs, reward), 1)                                
                l1_dist = torch.abs(probs.unsqueeze(1) - probs.unsqueeze(0)).mean(2)
                sum_reward += l1_dist.max().item()
            elif mode == 'exploit':
                sum_reward += reward.mean().item()
            actions.append(node.action)
        terminal = ((orig_level + len(actions)) == max_horizon + 1)
        # expansion
        if not terminal:
            node.children = [Node(node, a) for a in range(0, n_actions)]
            random.shuffle(node.children)

        # playout
        while not terminal:
            action = random.randint(0, n_actions - 1)
            action_one_hot = torch.tensor([i == action for i in range(n_actions)]).float()
            action_one_hot = action_one_hot.unsqueeze(0).repeat(n_ensemble*n_samples, 1).cuda()
            horizon = torch.tensor([i == (orig_level + len(actions)) for i in range(max_horizon)]).float().unsqueeze(0).repeat(n_ensemble*n_samples, 1).cuda()
            state_action = torch.cat((samples, action_one_hot, horizon), 1).cuda()
#            s_probs, x_probs, reward, samples = model(state_action, n_samples=1)
            x_probs, reward, samples = model(state_action, n_samples=1)
            if mode == 'explore':
                probs = samples.view(n_ensemble, n_samples, -1).mean(1)
                reward = reward.view(n_ensemble, n_samples).mean(1).unsqueeze(1)
                probs = torch.cat((probs, reward), 1)                
                l1_dist = torch.abs(probs.unsqueeze(1) - probs.unsqueeze(0)).mean(2)
                sum_reward += l1_dist.max().item()
            elif mode == 'exploit':
                sum_reward += reward.mean().item()
            actions.append(action)
            terminal = ((orig_level + len(actions)) == max_horizon + 1)

            if len(actions) > max_horizon + 1:
                pdb.set_trace()
                sum_reward -= 100
                break

        # remember best
        if best_reward < sum_reward:
            best_reward = sum_reward
            best_actions = actions

        # backpropagate
        while node:
            node.visits += 1
            node.value += sum_reward
            node = node.parent
    return best_actions
