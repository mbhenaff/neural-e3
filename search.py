import torch, math, random, numpy, pdb, time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils

import models
from models import *
import copy

class UniformExploration:

    def __init__(self, config):
        self.num_actions = config.n_actions

    def search(self, state, model, goal=None, eps=0, max_steps=0):
        a = random.randint(0, self.num_actions - 1)
        return [a], None, None, 0, None

            
class ParticleSearch2:
    def __init__(self, config):
        self.config = config
        pass

    def plot2d(self, vertices, k):
        plt.close()
        plt.scatter(vertices[:, 0].cpu().numpy(), vertices[:, 1].cpu().numpy(), color='black')
        plt.scatter(vertices[k, 0].cpu().numpy(), vertices[k, 1].cpu().numpy(), color='red')
        plt.scatter(vertices[0, 0].cpu().numpy(), vertices[0, 1].cpu().numpy(), color='blue')
        plt.show()
        
    def search(self, state, model, goal='explore', eps=0.9, max_steps=5000):
        state = state.view(-1, model.config.n_input_frames,model.config.n_input_channels, model.config.height, model.config.width)
        assert(state.size(0) == 1)
        paths, expanded_vertices = [], []
        returns_r, returns_u = [], []
        priorities = []

        # vertices will be [n_vertices x n_particles x state_dim]
        if self.config.phi != 'learned':
            state = state.float().cuda().view(1, model.config.edim)
            vertices = state.detach().unsqueeze(1).repeat(1, model.config.n_ensemble, 1)
        else:
            vertices = state.detach().unsqueeze(1).repeat(1, model.config.n_ensemble, 1, 1, 1, 1)

        vertices_mean = torch.mean(vertices, 1).view(1, -1)
            
        paths.append([])
        returns_r.append(0.0)
        returns_u.append(0.0)
        priorities.append(math.inf)
        # these will store uncertainty values and actions
        u_list, a_list = [], []
        model.eval()        
        exit_cond = 0
        done = False

        sim, sim_new = None, None
        for i in range(max_steps):
            # compute similarity matrix between vertices
            # for each vertex, get the closeness of its nearest neighbor (excluding itself)
            n_vertices = vertices.size(0)

            if n_vertices % 500 == 0:
                print(f'[searching...expanded {n_vertices} vertices]')
            if n_vertices > self.config.max_planning_steps: break
            torch.cuda.empty_cache()
            # top priority vertex
            k = torch.max(torch.tensor(priorities), 0)[1]
            priorities[k] = -math.inf
            vertex, path, ret_r, ret_u = vertices[k], paths[k], returns_r[k], returns_u[k]
            actions = numpy.random.permutation(model.config.n_actions)
            for a in actions:
                action = utils.one_hot(a, model.config.n_actions)
                model.train()
                s_pred, r_pred = model.forward_model.forward_all(vertex.cuda(), action.cuda())
                s_pred = s_pred.squeeze().detach()
                u = utils.compute_uncertainty(s_pred).item()
                model.eval()
                u_list.append(u)
                a_list.append(a)
                r_pred = r_pred.mean().item()
                ret_u_pred = u + ret_u
                ret_r_pred = r_pred + ret_r
                path_length = len(path)+1
                if goal == 'explore' and ret_u_pred/path_length > eps:
                    print(f'found (u={u:.6f} > eps={eps:.6f}) after {len(u_list)} tries, a=' + str(path + [a]) + f', |a|={len(path)+1}')
                    exit_cond = 1
                    done = True
                elif goal == 'exploit' and ret_r_pred/path_length > model.config.rmax:
                    print(f'found (r={ret_r_pred:.6f}) after {len(u_list)} tries, a=' + str(path + [a]) + f', |a|={len(path)+1}')
                    exit_cond = 2
                    done = True
                if done:                    
                    path = path + [a]
                    # compute state sequence
                    if not self.config.input_type == 'image':
                        s = vertices[0].unsqueeze(0)
                    else:
                        s = vertices[0]
                    s_seq, u_seq = model.rollout(s.cuda(), path)
                    return path, s_seq, u_seq, exit_cond, vertices.detach()
                else:
                    if self.config.input_type == 'image':
                        s_pred_mean = s_pred.mean(0).view(1, -1)
                        dist = ((vertices_mean.cuda() - s_pred_mean)**2).mean(1)
                        priorities.append(dist.min().item())
                        torch.cuda.empty_cache()
                        vertices = torch.cat((vertices, s_pred.unsqueeze(1).unsqueeze(0).cpu()), 0)
                        vertices_mean = torch.cat((vertices_mean, s_pred_mean.cpu()), 0)
                    else:
                        s_pred_mean = s_pred.mean(0).view(1, -1)
                        dist = ((vertices_mean.cuda() - s_pred_mean)**2).mean(1)
                        priorities.append(dist.min().item())
                        torch.cuda.empty_cache()                        
                        vertices = torch.cat((vertices, s_pred.unsqueeze(0)), 0)
                    paths.append(path + [a])
                    returns_r.append(ret_r_pred)
                    returns_u.append(ret_u_pred)

                    
        path_lengths = torch.tensor([len(p) for p in paths]).float()
        returns_u[0]=-math.inf
        returns_r[0]=-math.inf
        if goal == 'explore':
            returns_u = torch.tensor(returns_u) / path_lengths
            k = int(returns_u.numel()*0.01)
            max_u_indx = random.choice(torch.topk(returns_u, k)[1])
            max_u = returns_u[max_u_indx]            
            path = paths[max_u_indx]
            if len(path) == 0:
                path = [random.randint(0, model.config.n_actions-1)]            
            print(f'returning (u={max_u:.6f} < eps={eps:.6f}) after {max_steps} tries, a=' + str(path) + f' |a|={len(path)+1}')
        elif goal == 'exploit':
            returns_r = torch.tensor(returns_r) / path_lengths
            k = int(returns_r.numel()*0.01)
            max_r_indx = random.choice(torch.topk(returns_r, k)[1])
            max_r = returns_r[max_r_indx]
            path = paths[max_r_indx]
            if len(path) == 0:
                path = [random.randint(0, model.config.n_actions-1)]
            print(f'returning (r={max_r:.6f} < eps={model.config.rmax:.6f}) after {max_steps} tries, a=' + str(path) + f' |a|={len(path)+1}')
        return path, None, None, exit_cond, vertices.detach()
