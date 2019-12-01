#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import datetime
import torch
import time
import pdb, pickle, os
from .torch_utils import *
from pathlib import Path

def logtxt(fname, s, date=True):
    if not os.path.isdir(os.path.dirname(fname)):
        os.system(f'mkdir -p {os.path.dirname(fname)}')
    f = open(fname, 'a')
    if date:
        f.write(f'{str(datetime.datetime.now())}: {s}\n')
    else:
        f.write(f'{s}\n')
    f.close()


def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    logtxt(agent.logger.log_dir + '.csv', 'episodes, mean 10 episode reward', date=False)
    while True:
        total_episodes = agent.total_steps / config.horizon
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
        if agent.total_steps > 0 and config.log_interval and not total_episodes % config.log_interval:
#            mean_reward = agent.cumulative_reward / (agent.total_steps / config.horizon)
            running_mean_reward_10_ep = np.mean(agent.ep_rewards[-10:])
            log_string = 'steps %d, episodes %d, %.2f steps/s, total rew %.2f, mean rew (10 ep) %.2f' % (agent.total_steps, total_episodes, config.log_interval / (time.time() - t0), agent.cumulative_reward, running_mean_reward_10_ep)
            agent.logger.info(log_string)
            logtxt(agent.logger.log_dir + '.txt', log_string)
            t0 = time.time()
            logtxt(agent.logger.log_dir + '.csv', f'{total_episodes},{running_mean_reward_10_ep}', date=False)
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            break
        agent.step()
        agent.switch_task()
    '''
    trajs = None
    try:
        trajs = [agent.actor._task.env.envs[i].trajectories for i in range(config.num_workers)]
    except:
        try:
            trajs = [agent.task.env.envs[i].trajectories for i in range(config.num_workers)]
        except:
            pass
    for i, traj in enumerate(trajs):
        traj_file = agent.logger.log_dir + f'.env{i}.traj'
        pickle.dump(traj, open(traj_file, 'wb'))
    '''


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def generate_tag(params):
    if 'tag' in params.keys():
        return
    game = params['game']
    params.setdefault('run', 0)
    run = params['run']
    del params['game']
    del params['run']
    str = ['%s_%s' % (k, v) for k, v in sorted(params.items())]
    tag = '%s-%s-run-%d' % (game, '-'.join(str), run)
    params['tag'] = tag
    params['game'] = game
    params['run'] = run


def translate(pattern):
    groups = pattern.split('.')
    pattern = ('\.').join(groups)
    return pattern


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
