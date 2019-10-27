import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb, random
import models, mcts

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'level'))


        
        

class Uniform(object):
    def get_name(self):
        return "uniform"

    def __init__(self,actions,params={}):
        self.params = params
        ## NOTE: all environments have max_reward 1
        self.max_reward = 1
        
        if 'horizon' in params.keys():
            self.horizon=params['horizon']
        
        self.num_actions = actions
        print("[Uniform] Initialized")

    def select_action(self, x):
        action = random.randint(0, self.num_actions-1)
        return (action)

    def get_value(self,x):
        pass

    def save_transition(self, state, action, reward, next_state, level):
        pass

    

    def finish_episode(self):
        pass

    def state_to_str(self,x):
        return("".join([str(z) for z in x]))
