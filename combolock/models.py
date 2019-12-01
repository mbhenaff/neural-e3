import random, pdb, math, copy, numpy
import torch
import torch.nn as nn
import torch.nn.functional as F



class DQN(nn.Module):
    def __init__(self, params):
        super(DQN, self).__init__()
        self.params = params
        self.n_inputs = 3 + params['dimension'] + params['horizon']
        self.q_network = nn.Sequential(
            nn.Linear(self.n_inputs, params['n_hidden']),
            nn.ReLU(),
            nn.Linear(params['n_hidden'], params['n_hidden']),
            nn.ReLU(),
            nn.Linear(params['n_hidden'], params['n_actions'])
        )

        self.q_network2 = nn.Sequential(
            nn.Linear(self.n_inputs, params['n_hidden']),
            nn.ReLU(),
            nn.Linear(params['n_hidden'], params['n_hidden']),
            nn.ReLU(),
            nn.Linear(params['n_hidden'], params['n_actions'])
        )

        self.sync_networks()
        self.batch_indices = torch.arange(0, self.params['batch_size']).long()

    def sync_networks(self):
        self.q_network2.load_state_dict(self.q_network.state_dict())

                
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
            q_next = q_next*0.99*(1-terminals)
            q_next.add_(rewards)
            loss = F.smooth_l1_loss(q, q_next)
            return q, loss




class ForwardModel(nn.Module):

    def __init__(self, params):
        super(ForwardModel, self).__init__()
        self.params = params
        self.model = nn.Sequential(EnsembleLinearGPU(3 + params['dimension'] + params['n_actions'] + params['horizon'], params['n_hidden'], params['n_ensemble']),
                                   nn.ReLU(),
                                   EnsembleLinearGPU(params['n_hidden'], params['n_hidden'], params['n_ensemble']),
                                   nn.ReLU(),
                                   EnsembleLinearGPU(params['n_hidden'], params['dimension'] + 3 + 1, params['n_ensemble']),
                                   )

        if params['anchor'] == 1:
            self.model2 = nn.Sequential(EnsembleLinearGPU(3 + params['dimension'] + params['n_actions'] + params['horizon'], params['n_hidden'], params['n_ensemble']),
                                       nn.ReLU(),
                                       EnsembleLinearGPU(params['n_hidden'], params['n_hidden'], params['n_ensemble']),
                                       nn.ReLU(),
                                       EnsembleLinearGPU(params['n_hidden'], params['dimension'] + 3 + 1, params['n_ensemble']),
            )


    def forward(self, x, n_samples=0):
        batch_size = x.size(0)
        out = self.model(x)
        out = out.view(out.size(0)*out.size(1), -1)

        if self.params['anchor'] == 1:
            out2 = self.model(x).detach()
            out2 = out2.view(out2.size(0)*out2.size(1), -1)
            out = out + out2
            
        x_next = torch.sigmoid(out[:, :-1])
        r = out[:, -1]        
        if n_samples > 0:
            z = torch.rand(n_samples, x_next.size(0), x_next.size(1)).cuda()
            x_next_rep = x_next.unsqueeze(0).repeat(n_samples, 1, 1)
            x_next_samples = (z < x_next_rep).float()
            x_next_samples = x_next_samples.view(n_samples, batch_size, -1)
            samples = x_next_samples.squeeze()
        else:
            samples = None
        return x_next, r, samples



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
            weight.data.copy_(w.weight.data)
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

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, models={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.n_ensemble
        )
