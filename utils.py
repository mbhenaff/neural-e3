import torch, pdb, math, numpy, os, atexit, logging, time
from datetime import datetime
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from multiprocessing import Process, Queue


class Timer:
    def __init__(self):
        self.t = time.time()

    def tic(self):
        t0 = self.t
        self.t = time.time()
        return self.t - t0
    

class Tensorboard:

    def __init__(self, experiment, log_dir="tensorboard_logs"):
        experiment_name = experiment.split("/")[-1]
        save_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(save_dir)
        self.index_dict = dict()

    def log_scalar(self, name, value, index=-1):
        if index == -1:
            if name in self.index_dict:
                self.index_dict[name] += 1
                index = self.index_dict[name]
            else:
                self.index_dict[name] = 1
                index = 1
        self.writer.add_scalar(name, value, index)

    def log_histogram(self, name, value, bins, index=-1):
        if index == -1:
            if name in self.index_dict:
                self.index_dict[name] += 1
                index = self.index_dict[name]
            else:
                self.index_dict[name] = 1
                index = 1
        self.writer.add_histogram(name, value, index, bins)



def compute_uncertainty(s_pred, method='max'):
    if method == 'max':
        diff = torch.sum((s_pred.squeeze().unsqueeze(0) - s_pred.unsqueeze(1))**2, 2)
        u = diff.max()
    elif method == 'var':
        u = torch.var(s_pred, 0).sum()
    return u
        
        
        
def one_hot(i, d, cuda=True, unsqueeze=True):
    x = torch.zeros(d)
    x[i] = 1
    if cuda: x = x.cuda()
    if unsqueeze:
        x = x.unsqueeze(0)
    return x

def grad_norm(model):
    norm = 0
    for w in model.parameters():
        if w.grad is not None:
            norm += w.grad.norm().detach().item()
    return norm

def count_parameters(model):
    cnt = 0
    for p in model.parameters():
        cnt = cnt + p.numel()
    return cnt
    

def precision_at_k(scores, labels, k=5, dim=1):
    b = scores.size(0)
    topk = torch.topk(scores, k=k, dim=dim, largest=True)[1].float()
    matches = labels.view(-1, 1).expand(b, k) == topk
    return matches.float().sum().item() / b
    

def scatter(data, k=None, size=(80, 80, 3)):
    x = data[:, 0]
    y = data[:, 1]
    # convert to pixel coordinates
    x = x - x.min()
    x = x / x.max()
    x = x * (size[0]-1)
    y = y - y.min()
    y = y / y.max()
    y = y * (size[1]-1)
    n = data.shape[0]
    img = numpy.zeros(size)
    for i in range(n):
        img[int(round(x[i]))][int(round(y[i]))][0] = 0.5
        img[int(round(x[i]))][int(round(y[i]))][1] = 0.5
        img[int(round(x[i]))][int(round(y[i]))][2] = 0.5

    if k is not None:
        img[int(round(x[k]))][int(round(y[k]))][0] = 1
        img[int(round(x[k]))][int(round(y[k]))][1] = 0
        img[int(round(x[k]))][int(round(y[k]))][2] = 0
    return img


# simple logger

class SimpleLogger():
    
    def __init__(self, fname):
        self.fname = fname
        if not os.path.isdir(os.path.dirname(fname)):
            os.system(f'mkdir -p {os.path.dirname(fname)}')

    def log(self, s, date=True):
        f = open(self.fname, 'a')
        if date:
            s = f'{str(datetime.now())}: {s}'
            f.write(s + '\n')
        else:
            s = f'{s}'
            f.write(s + '\n')
        print(s)
        f.close()


        
    
def logtxt(fname, s, date=True):
    if not os.path.isdir(os.path.dirname(fname)):
        os.system(f'mkdir -p {os.path.dirname(fname)}')
    f = open(fname, 'a')
    if date:
        s = f'{str(datetime.now())}: {s}'
        f.write(s + '\n')
    else:
        s = f'{s}'
        f.write(s + '\n')
    print(s)
    f.close()
