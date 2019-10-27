import os, pdb, time, torch, random, numpy, scipy, datetime, math
import sys, logging, traceback, argparse, copy
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

import environment, agents, plotting, models, utils, importlib, arguments
from utils import Tensorboard


def main():

    config, experiment_name = arguments.get_args()

    # Set seed
    random.seed(config.seed)
    numpy.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    experiment = f'{config.results_dir}/{experiment_name}/'
    print("EXPERIMENT NAME: ", experiment_name)

    # Create the experiment folder and logger
    if not os.path.exists(experiment):
        os.makedirs(experiment)
    logger = utils.SimpleLogger(f'{experiment}/log.txt')
    
    # Copy source code
    srcpath = experiment + '/src/'
    if not os.path.exists(srcpath):
        os.makedirs(srcpath)
        os.system(f'cp *.py {srcpath}')
        
    # Define log settings
    log_path = experiment + '/train_baseline.log'

    # Create agent and environment
    env = environment.EnvironmentWrapper(config)
    agent = agents.Agent(config)
    if config.cuda == 1: agent = agent.cuda()
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate)
    agent.dqn = models.DQN(config).cuda()
    optimizer_dqn = optim.Adam(agent.dqn.parameters(), lr=config.dqn_learning_rate)
    agent.best_dqn_params = agent.dqn.state_dict()
    keep_training_dqn = True
    print(f'# parameters: {utils.count_parameters(agent)}')

    # Load checkpoint if one exists
    if os.path.isfile(experiment + '/agent.pth'):
        print(f'[loading checkpoint from {experiment}]')
        checkpoint = torch.load(experiment + '/agent.pth')
        agent.load_state_dict(checkpoint['agent'].state_dict())
        optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
        agent.replay_memory = checkpoint['agent'].replay_memory
        epoch = checkpoint['ep'] + 1
        perf = torch.load(experiment + '/perf.pth')
        print(f'[resuming at epoch {epoch}]')
    else:
        epoch = 0
        perf = {'losses': [], 'metrics': [], 'rewards': []}
    tensorboard = Tensorboard(config.results_dir + f'/tensorboard/{experiment_name}', log_dir=config.results_dir + '/tensorboard_logs/')
    
    best_exploit_perf = -math.inf
    dqn_epochs_completed = 0

    # Start algorithm
    phase = 'explore'
    while epoch < 200:

        if epoch < config.n_exploration_epochs:
            #### Explore phase
            phase = 'explore'
            agent.eval()
            exploration_policy = 'random' if epoch == 0 else config.exploration_policy
            ep_reward, ep_length = agent.act(env, 'train', config, policy = exploration_policy, goal='explore')
            logger.log(f'EXPLORE PHASE | mean reward: {ep_reward}, mean episode length: {ep_length}')
            if config.test == 1:
                agent.act(env, 'test', config, policy = exploration_policy, goal='explore')
            # train the models
            for i in range(config.n_training_epochs):
                if (i < config.n_training_epochs - 1):
                    split = 'train'; agent.train()
                else:
                    split = 'test'; agent.eval()
                losses, log_string = agent.train_model(split, 'explore', optimizer, config, tensorboard, update=(split=='train'))
                logger.log(f'TRAINING MODEL | ep {epoch}/{i} | {log_string}')
        else:
            #### Exploit phase
            if 'maze' in config.env:
                # just do search
                ep_reward, ep_length = agent.act(env, 'train', config, policy = 'particle2', goal='exploit', n_episodes = config.n_trajectories)
                logger.log(f'EXPLOIT PHASE: epoch {epoch}, mean reward: {ep_reward}, mean episode length: {ep_length}')
            else:
                if phase == 'explore': 
                    # this is our first time exploiting - train the DQN for a while
                    phase = 'exploit'
                    agent.train_policy_dqn('train', 'explore', optimizer_dqn, config, n_updates=config.dqn_model_updates, logger=logger)
                    agent.best_dqn_params = copy.deepcopy(agent.dqn.state_dict())
                    keep_training_dqn = True
                else:
                    # act in the environment
                    if keep_training_dqn:
                        agent.train_policy_dqn('train', 'explore', optimizer_dqn, config, n_updates=25000, logger=logger)
                    ep_reward, ep_length = agent.act(env, 'test', config, policy='dqn', n_episodes=config.dqn_eval_ep)
                    logger.log(f'EXPLOIT PHASE: epoch {epoch}, mean reward: {ep_reward}, mean episode length: {ep_length}, DQN training: {keep_training_dqn}')
                    if keep_training_dqn:
                        if ep_reward >= best_exploit_perf or config.checkpoint_dqn == 0:
                            best_exploit_perf = ep_reward
                            agent.best_dqn_params = copy.deepcopy(agent.dqn.state_dict()) # TODO clone!!!!
                        else:
                            agent.dqn.load_state_dict(agent.best_dqn_params)
                            keep_training_dqn = False
            
        perf['epoch'] = epoch
        perf['rewards'].append(ep_reward)
        torch.save(perf, f'{experiment}/perf.pth')
        torch.save({'agent': agent, 'optimizer': optimizer, 'ep': epoch}, f'{experiment}/agent.epoch{epoch}.pth')
        torch.save({'agent': agent, 'optimizer': optimizer, 'ep': epoch}, f'{experiment}/agent.pth')
        torch.save(agent.replay_memory, f'{experiment}/replay_memory.pth')
        epoch += 1

                                        

if __name__ == "__main__":
    main()
