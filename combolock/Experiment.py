import numpy as np
import sys, os
import gym
import Environments, Params
import OracleQ, Decoding, QLearning, NeuralE3, Uniform
import argparse
import torch
import random
import pdb



def logtxt(fname, s, date=False):
    if not os.path.isdir(os.path.dirname(fname)):
        os.system(f'mkdir -p {os.path.dirname(fname)}')
    f = open(fname, 'a')
    if date:
        f.write(f'{str(datetime.now())}: {s}\n')
    else:
        f.write(f'{s}\n')
    f.close()


torch.set_default_tensor_type(torch.FloatTensor)

def parse_environment_params(args):
    ep_dict = {'horizon': args.horizon,
               'dimension': args.dimension,
               'antishaping': args.antishaping,
               'tabular': args.tabular}
    if args.env_param_1 is not None:
        ep_dict['switch'] = float(args.env_param_1)
    if args.env_param_2 is not None:
        if args.env_param_2 == 'None':
            ep_dict['noise'] = None
        else:
            ep_dict['noise'] = float(args.env_param_2)
    return (ep_dict)

def get_env(name, args):
    env = gym.make(name)
    ep_dict = parse_environment_params(args)
    env.seed(args.seed+args.iteration*31)
    env.init(env_config=ep_dict)
    return(env)

def get_alg(name, args, env):
    if name == "oracleq":
        alg_dict = {'horizon': args.horizon,
                    'alpha': args.lr,
                    'conf': args.conf }
        alg = OracleQ.OracleQ(env.action_space.n, params=alg_dict)
    elif name == "neural-e3":
        alg_dict = {'horizon': args.horizon,
                    'dimension': args.dimension,
                    'n_hidden': args.n_hidden,
                    'n_ensemble': args.n_ensemble,
                    'n_playouts': args.n_playouts,
                    'n_samples': args.n_samples,
                    'n_model_updates': args.n_model_updates,
                    'lr': args.lr,
                    'batch_size': 100, 
                    'n_actions': env.action_space.n,
                    'conf': args.conf }
        alg = NeuralE3.NeuralE3(env.action_space.n, params=alg_dict)
    elif name == 'decoding':
        alg_dict = {'horizon': env.horizon,
                    'model_type': args.model_type,
                    'n': args.n,
                    'num_cluster': args.num_cluster}
        alg = Decoding.Decoding(env.observation_space.n, env.action_space.n,params=alg_dict)
    elif name == 'uniform':
        alg_dict = {'horizon': env.horizon,
                    'model_type': args.model_type,
                    'n_actions': env.action_space.n,
                    'n': args.n,
                    'num_cluster': args.num_cluster}
        alg = Uniform.Uniform(env.action_space.n, params=alg_dict)
    elif name=='qlearning':
        assert args.tabular, "[EXPERIMENT] Must run QLearning in tabular mode"
        alg_dict = {
            'alpha': float(args.lr),
            'epsfrac': float(args.epsfrac),
            'num_episodes': int(args.episodes)}
        alg = QLearning.QLearning(env.action_space.n, params=alg_dict)
    return (alg)

def parse_args():
    parser = argparse.ArgumentParser(description='StateDecoding Experiments')
    parser.add_argument('--seed', type=int, default=368, metavar='N',
                        help='random seed (default: 367)')
    parser.add_argument('--system', type=str, default='local', metavar='N',
                        help='local | philly')
    parser.add_argument('--results_dir', type=str, default='results_lock')
    parser.add_argument('--iteration', type=int, default=1,
                        help="Which replicate number")
    parser.add_argument('--env', type=str, default="Lock-v0",
                        help='Environment', choices=["Lock-v0", "Lock-v1"])
    parser.add_argument('--horizon', type=int, default=4,
                        help='Horizon')
    parser.add_argument('--dimension', type=int, default=5,
                        help='Dimension')
    parser.add_argument('--tabular', type=bool, default=False,
                        help='Make environment tabular')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Training Episodes')
    parser.add_argument('--episodes_exp_ratio', type=int, default=100,
                        help='Exploration Episodes divided by horizon')
    parser.add_argument('--n_ensemble', type=int, default=10,
                        help='Ensemble size')
    parser.add_argument('--n_playouts', type=int, default=100,
                        help='playouts')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='samples')
    parser.add_argument('--n_model_updates', type=int, default=100,
                        help='updates')
    parser.add_argument('--n_hidden', type=int, default=50,
                        help='hiddens')
    parser.add_argument('--env_param_1', type=str,
                        help='Additional Environment Parameters (Switching prob)', default=None)
    parser.add_argument('--env_param_2', type=str,
                        help='Additional Environment Parameters (Feature noise)', default=None)
    parser.add_argument('--alg', type=str, default='neural-e3',
                        help='Learning Algorithm', choices=["oracleq", "decoding", "qlearning", "neural-e3", "uniform"])
    parser.add_argument('--model_type', type=str, default='linear',
                        help='What model class for function approximation', choices=['linear', 'nn'])
    parser.add_argument('--lr', type=float,
                        help='Learning Rate for optimization-based algorithms', default=3e-2)
    parser.add_argument('--epsfrac', type=float,
                        help='Exploration fraction for Baseline DQN.', default=0.1)
    parser.add_argument('--conf', type=float,
                        help='Exploration Bonus Parameter for Oracle Q.', default=3e-2)
    parser.add_argument('--n', type=int, default = 200,
                        help="Data collection parameter for decoding algoithm.")
    parser.add_argument('--num_cluster', type=int, default = 3,
                        help="Num of hidden state parameter for decoding algoithm.")
    parser.add_argument('--antishaping', type=float, default = 0.0,
                        help="Antishaped Reward")
    args = parser.parse_args()
    return(args)

def train(env, alg, args):
    T = args.episodes
    if args.system == 'local' or args.system == 'gcr':
        args.results_dir = '/nycml/mihenaff/results/' + args.results_dir
    elif args.system == 'philly':
        args.results_dir = os.getenv('PT_OUTPUT_DIR') + '/'
    
    experiment_name = f'alg={args.alg}'
    experiment_name += f'-h={args.horizon}'
    experiment_name += f'-dim={args.dimension}'
    experiment_name += f'-lr={args.lr}'
    experiment_name += f'-playouts={args.n_playouts}'
    experiment_name += f'-samples={args.n_samples}'
    experiment_name += f'-nh={args.n_hidden}'
    experiment_name += f'-nup={args.n_model_updates}'
    experiment_name += f'-epexp={args.episodes_exp_ratio}'
    experiment_name += f'-seed={args.seed}'

    print(f'will save as {experiment_name}')
    experiment_name = args.results_dir + f'/{experiment_name}'

    if os.path.isfile(experiment_name + '/results.pth'):
        print(f'checkpoint found: {experiment_name}/results.pth')
        checkpoint = torch.load(experiment_name + '/results.pth')
        alg = checkpoint['alg']
        env = checkpoint['env']
        running_reward = checkpoint['running_reward']
        state_visitation = checkpoint['state_visitation']
        reward_vec = checkpoint['reward_vec']
        ep_rewards = checkpoint['ep_rewards']
        t = checkpoint['t']
        print(f'resuming at episode {t}')
    else:
        running_reward = 0
        reward_vec = []
        state_visitation = np.zeros((3, args.horizon))
        ep_rewards = []
        t = 1
    while t < T+1:
        state = env.reset()
        if t <= args.episodes_exp_ratio * args.horizon:
            alg.mode = 'explore'
        else:
            alg.mode = 'exploit'
        done = False
        level = 0
        ep_reward = 0
        while not done:
            state_visitation[:, level] += state[:3]
            if args.alg == 'neural-e3':
                action = alg.select_action(state, level)
            else:
                action = alg.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if args.alg in ['neural-e3', 'uniform']:
                alg.save_transition(state, action, reward, next_state, level)
            else:
                alg.save_transition(state, action, reward, next_state)
            state = next_state
            running_reward += reward
            ep_reward += reward
            level += 1
        alg.finish_episode()
        ep_rewards.append(ep_reward)
        log_string = f'ep {t} ({alg.mode}) | reward: {ep_reward}, running reward: {running_reward}'
        logtxt(f'{experiment_name}/output.log', log_string)
        logtxt(f'{experiment_name}/output.log', np.array2string(state_visitation))
        torch.save(ep_rewards, f'{experiment_name}/perf.pth')
        torch.save({'alg': alg, 'env': env}, f'{experiment_name}/alg_ep{t}.pth')
        checkpoint_data = {'alg': alg,
                           'env': env,
                           'running_reward': running_reward,
                           'state_visitation': state_visitation,
                           'reward_vec': reward_vec,
                           'ep_rewards': ep_rewards,
                           't': t
                           }
        torch.save(checkpoint_data, f'{experiment_name}/results.pth')
        print(log_string)
        print(state_visitation)
        if t % 100 == 0:
            reward_vec.append(running_reward)
        if t % 1000 == 0:
            log_string = "[EXPERIMENT] Episode %d Completed. Average reward: %0.2f" % (t, running_reward/t)
            logtxt(experiment_name + '/output.log', log_string)
            print(log_string)
        t += 1
            
    return (reward_vec)

def main(args):
    random.seed(args.seed+args.iteration*29)
    np.random.seed(args.seed+args.iteration*29)

    import torch
    torch.manual_seed(args.seed+args.iteration*37)

    env = get_env(args.env, args)
    alg = get_alg(args.alg, args, env)


    P = Params.Params(vars(args))
    fname = P.get_output_file_name()
    reward_vec = train(env,alg,args)

    print("[EXPERIMENT] Learning completed")
    f = open(fname,'w')
    f.write(",".join([str(z) for z in reward_vec]))
    f.write("\n")
    f.close()
    print("[EXPERIMENT] Done")
    return None

if __name__=='__main__':
    Args = parse_args()
    print(Args)
    main(Args)
