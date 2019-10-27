import random, pdb, math, copy, numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from experience import Rollout
import search, utils


class ForwardModelImageEnsembleSmall(nn.Module):

    def __init__(self, config, vae=False):
        super(ForwardModelImageEnsembleSmall, self).__init__()
        self.config = config
        n_channels = config.n_input_channels
        n_frames = config.n_input_frames
        n_feats = config.n_feature_maps
        n_models = config.n_ensemble
        self.n_channels = n_channels
        self.n_models = n_models
        self.n_feats = n_feats
        self.n_frames = n_frames
        
        self.conv1 = nn.Conv2d(n_models*n_channels*n_frames, n_models*n_feats, 3, 1, 1, groups=n_models)
        self.deconv1 = nn.ConvTranspose2d(n_models*n_feats, n_models*n_feats, 3, 1, 1, groups=n_models)
        self.deconv2 = nn.ConvTranspose2d(n_models*n_feats, n_models*n_channels, 3, 1, 1, groups=n_models)

        self.action_encoder = EnsembleLinearGPU(self.config.n_actions, self.n_feats, n_models)

        self.reward_predictor_conv = nn.Sequential(nn.Conv2d(n_models*n_feats, n_models*n_feats, 3, 2, 1, groups=n_models), 
                                                   nn.ReLU(),
                                                   nn.Conv2d(n_models*n_feats, n_models*n_feats, 3, 2, 1, groups=n_models),
                                                   nn.ReLU(), 
                                                   nn.Conv2d(n_models*n_feats, n_models*n_feats, 3, 2, 1, groups=n_models),
                                                   nn.ReLU()
                                                   )
        

        # dry run to get FC layer sizes
        bsize = 8
        state = torch.randn(bsize, self.n_models*n_feats, self.config.height, self.config.width)
        h = self.reward_predictor_conv(state).view(bsize*self.n_models, -1)
        self.reward_predictor_fc = nn.Sequential(EnsembleLinearGPU(h.size(1), n_feats, n_models),
                                                 nn.ReLU(),
                                                 EnsembleLinearGPU(n_feats, 1, n_models)
                                                )
        

        

    def forward(self, state, action):
        nsamples = action.size(0)
        assert(nsamples % self.n_models == 0)
        bsize = int(nsamples/self.n_models)
        state = state.contiguous()
        state = state.view(bsize, self.n_models*self.n_channels*self.n_frames, self.config.height, self.config.width).contiguous()

        e1 = F.relu(self.conv1(state))
        aemb = self.action_encoder(action)
        aemb = aemb.view(bsize, self.n_models*self.n_feats, 1, 1)
        
        e2 = e1 + aemb
        d2 = F.relu(self.deconv1(e2))
        d1 = self.deconv2(d2)
        hr = self.reward_predictor_conv(e2)
        r_pred = self.reward_predictor_fc(hr.view(bsize*self.n_models, -1)).view(bsize*self.n_models)
        state = state.view(bsize*self.n_models, self.n_channels*self.n_frames, self.config.height, self.config.width).contiguous()        
        out = state + d1.view(bsize*self.n_models, self.n_channels, self.config.height, self.config.width)
        return out, r_pred

    def forward_all(self, phi, action, particles=True):
        s_pred, r_pred  = self.forward(phi, action.repeat(phi.size(0), 1))
        return s_pred, r_pred



    
class ForwardModel(nn.Module):
    def __init__(self, config):
        super(ForwardModel, self).__init__()
        self.config = config

        self.network1 = nn.Sequential(
            nn.Linear(config.edim, config.n_hidden),
            nn.LayerNorm(config.n_hidden), 
            nn.Dropout(p=config.p_dropout),
            nn.LeakyReLU(0.2),
            nn.Linear(config.n_hidden, config.n_hidden)
        )

        self.network2 = nn.Sequential(
            nn.Linear(config.n_hidden, config.n_hidden),
            nn.LayerNorm(config.n_hidden), 
            nn.Dropout(p=config.p_dropout),
            nn.LeakyReLU(0.2),
            nn.Linear(config.n_hidden, config.n_hidden),
            nn.LayerNorm(config.n_hidden), 
            nn.Dropout(p=config.p_dropout),
            nn.LeakyReLU(0.2)
        )

        self.final_layer = nn.Linear(config.n_hidden, config.edim + 1)
        self.action_encoder = nn.Linear(config.n_actions, config.n_hidden)

        
    def forward(self, phi, action):
        phi = phi.squeeze()
        h = self.network1(phi)
        a = self.action_encoder(action)
        if self.config.a_combine == 'add':
            h = h + a
        elif self.config.a_combine == 'mult':
            h = h * a
        else:
            return ValueError
        h_final = self.network2(h)
        h = self.final_layer(h_final)
        phi_next = phi + h[:, :self.config.edim]
        r_pred = h[:, self.config.edim]
        return phi_next, r_pred, h_final


# ensemble parallelized for GPU
class EnsembleLinearGPU(nn.Module):
    def __init__(self, in_features, out_features, n_ensemble, bias=True):
        super(EnsembleLinearGPU, self).__init__()    
        self.in_features = in_features
        self.out_features = out_features
        self.n_ensemble = n_ensemble
        self.bias = bias
        self.weights = nn.Parameter(torch.Tensor(n_ensemble, out_features, in_features))
        if bias:
            self.biases  = nn.Parameter(torch.Tensor(n_ensemble, out_features))
        else:
            self.register_parameter('biases', None)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            w = nn.Linear(self.in_features, self.out_features)
#            weight.data.copy_(w.weight.data)
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.biases is not None:
            for bias in self.biases:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights[0])
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, inputs):
        # check input sizes
        if inputs.dim() == 3:
            # assuming size is [n_ensemble x batch_size x features]
            assert(inputs.size(0) == self.n_ensemble and inputs.size(2) == self.in_features)
        elif inputs.dim() == 2:
            n_samples, n_features = inputs.size(0), inputs.size(1)
            assert (n_samples % self.n_ensemble == 0 and n_features == self.in_features), [n_samples, self.n_ensemble, n_features, self.in_features]
            batch_size = int(n_samples / self.n_ensemble)
            inputs = inputs.view(self.n_ensemble, batch_size, n_features)

        # reshape to [n_ensemble x n_features x batch_size]
        inputs = inputs.permute(0, 2, 1)
        outputs = torch.bmm(self.weights, inputs)
        outputs = outputs
        if self.bias:
            outputs = outputs + self.biases.unsqueeze(2)
        # reshape to [n_ensemble x batch_size x n_features]
        outputs = outputs.permute(0, 2, 1).contiguous()
        return outputs


class ForwardModelEnsembleGPU(nn.Module):
    def __init__(self, config):
        super(ForwardModelEnsembleGPU, self).__init__()
        self.config = config

        self.network1 = nn.Sequential(
            EnsembleLinearGPU(config.edim, config.n_hidden, config.n_ensemble),
            nn.LayerNorm(config.n_hidden, elementwise_affine=False),
            nn.Dropout(p=config.p_dropout),
            nn.LeakyReLU(0.2),
            EnsembleLinearGPU(config.n_hidden, config.n_hidden, config.n_ensemble),
            nn.LayerNorm(config.n_hidden, elementwise_affine=False)
        )

        self.network2 = nn.Sequential(
            EnsembleLinearGPU(config.n_hidden, config.n_hidden, config.n_ensemble),
            nn.LayerNorm(config.n_hidden, elementwise_affine=False),
            nn.Dropout(p=config.p_dropout),
            nn.LeakyReLU(0.2),
            EnsembleLinearGPU(config.n_hidden, config.n_hidden, config.n_ensemble),
            nn.LayerNorm(config.n_hidden, elementwise_affine=False),
            nn.Dropout(p=config.p_dropout),
            nn.LeakyReLU(0.2),
            EnsembleLinearGPU(config.n_hidden, config.edim + 1, config.n_ensemble)                  )

        self.action_encoder = EnsembleLinearGPU(config.n_actions, config.n_hidden, config.n_ensemble)

        
    def forward(self, phi, action):
        phi = phi.squeeze()
        h = self.network1(phi)
        a = self.action_encoder(action)
        if self.config.a_combine == 'add':
            h = h + a
        elif self.config.a_combine == 'mult':
            h = h * a
        else:
            return ValueError
        h = self.network2(h)
        phi_next = phi + h[:, :, :self.config.edim].contiguous().view(phi.size())
        r_pred = h[:, :, self.config.edim]
        r_pred = r_pred.contiguous().view(-1)
        return phi_next, r_pred

    def forward_all(self, phi, action, particles=True):
        s_pred, r_pred  = self.forward(phi, action.repeat(phi.size(0), 1))
        return s_pred, r_pred


        

class DQN(nn.Module):
    def __init__(self, config):
        super(DQN, self).__init__()
        self.config = config
        self.q_network = nn.Sequential(
            nn.Linear(config.edim, config.n_hidden),
            nn.ReLU(),
            nn.Linear(config.n_hidden, config.n_hidden),
            nn.ReLU(),
            nn.Linear(config.n_hidden, config.n_actions)
        )

        self.q_network2 = nn.Sequential(
            nn.Linear(config.edim, config.n_hidden),
            nn.ReLU(),
            nn.Linear(config.n_hidden, config.n_hidden),
            nn.ReLU(),
            nn.Linear(config.n_hidden, config.n_actions)
        )

        self.sync_networks()
        self.batch_indices = torch.arange(0, self.config.batch_size).long()

    def sync_networks(self):
        self.q_network2.load_state_dict(self.q_network.state_dict())

    def forward_nstep(self, states, actions=None, next_states=None, rewards=None, terminals=None, forward_model=None, nsteps=3):
        bsize = states.size(0)
        q = self.q_network(states)
        if actions is None:
            return q, None
        else:
            q = q[self.batch_indices, actions]
            values = torch.zeros(bsize).cuda()
            values.add_(rewards)
            s = next_states
            for n in range(nsteps):
                a = torch.argmax(self.q_network(s), dim=-1)
                a = torch.stack([utils.one_hot(a_i, self.config.n_actions) for a_i in a]).squeeze()
                s_pred, r_pred, _ = forward_model.forward(s, a) # TODO could pass each sample through all models and average
                values = values + (self.config.gamma**(n+1))*r_pred
                s = s_pred.detach()
            a = torch.argmax(self.q_network(s), dim=-1)
            q_next = self.q_network2(s).detach()            
            q_next = q_next[self.batch_indices, a]
            values = values + self.config.gamma**(nsteps+1)*q_next
            loss = F.smooth_l1_loss(q, values.detach())
            return q, loss
                
                
            
        
    def forward(self, states, actions=None, next_states=None, rewards=None, terminals=None):
        q = self.q_network(states)
        if actions is None:
            return q, None
        else:
            q = q[self.batch_indices, actions]
            q_next = self.q_network2(next_states).detach()
            q_next = q_next
            # Double DQN
            best_actions = torch.argmax(self.q_network(next_states), dim=-1)
            q_next = q_next[self.batch_indices, best_actions]
            q_next = q_next*self.config.gamma*(1-terminals)
            q_next.add_(rewards)
            loss = F.smooth_l1_loss(q, q_next)
            return q, loss



