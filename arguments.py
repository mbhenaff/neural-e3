import torch
import argparse, os
import environment

def get_args():

    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument('-system', type=str, default='local')
    parser.add_argument('-env', type=str, default='mountaincar')
    parser.add_argument('-seed', type=int, default=12345)    
    parser.add_argument('-cuda', type=int, default=1)
    parser.add_argument('-results_dir', type=str, default='/e3/scratch/')
    parser.add_argument('-test', type=int, default=1)
    # forward model parameters
    parser.add_argument('-n_feature_maps', type=int, default=64)
    parser.add_argument('-n_hidden', type=int, default=64)
    parser.add_argument('-p_dropout', type=float, default=0.0)
    parser.add_argument('-n_ensemble', type=int, default=8)
    parser.add_argument('-loss', type=str, default='mse')
    parser.add_argument('-phi_layer_size', type=int, default=512)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-n_training_epochs', type=int, default=20)
    parser.add_argument('-n_model_updates', type=int, default=100)
    parser.add_argument('-T', type=int, default=20)
    parser.add_argument('-a_combine', type=str, default='mult')
    # E3 parameters
    parser.add_argument('-u_quantile', type=float, default=-1)
    parser.add_argument('-eps', type=float, default=10.0)
    parser.add_argument('-lambda_r', type=float, default=1.0)
    parser.add_argument('-exploration_policy', type=str, default='particle2')
    parser.add_argument('-n_trajectories', type=int, default=10)
    parser.add_argument('-gamma', type=float, default=1.0)
    parser.add_argument('-max_exploration_steps', type=int, default=200)
    parser.add_argument('-max_planning_steps', type=int, default=5000)
    parser.add_argument('-zeroact', type=int, default=0)
    parser.add_argument('-n_exploration_epochs', type=int, default=10)
    # DQN parameters
    parser.add_argument('-dqn_learning_rate', type=float, default=0.00003)
    parser.add_argument('-dqn_update_target_freq', type=int, default=5000)
    parser.add_argument('-dqn_grad_clip', type=float, default=1)
    parser.add_argument('-dqn_model_updates', type=int, default=750000)
    parser.add_argument('-dqn_pos_reward_prob', type=float, default=0.1)
    parser.add_argument('-checkpoint_dqn', type=int, default=1)
    parser.add_argument('-dqn_eval_ep', type=int, default=20)
    parser.add_argument('-rscale', type=float, default=1.0)
    config = parser.parse_args()

    config = environment.add_constants(config)

    if config.system == 'local' or config.system == 'gcr':
        config.results_dir = '/nycml/mihenaff/results/' + config.results_dir
    elif config.system == 'philly':
        config.results_dir = os.getenv('PT_OUTPUT_DIR') + config.results_dir

    # Define experiment name based on parameters
    experiment_name = f'env={config.env}'
    experiment_name += f'-pol={config.exploration_policy}'
    experiment_name += f'-l={config.loss}'
    experiment_name += f'-nhid={config.n_hidden}'
    experiment_name += f'-T={config.T}'
    experiment_name += f'-lr={config.learning_rate}'
    experiment_name += f'-ens={config.n_ensemble}'
    experiment_name += f'-ntraj={config.n_trajectories}'
    experiment_name += f'-nup={config.n_model_updates}'
    experiment_name += f'-mexp={config.n_exploration_epochs}'
    experiment_name += f'-rw={config.lambda_r}'
    experiment_name += f'-maxp={config.max_planning_steps}'
    
    if config.u_quantile == -1:
        experiment_name += f'-eps={config.eps}'
    else:
        experiment_name += f'-uquant={config.u_quantile}'    

    if config.input_type == 'image':
        experiment_name += f'-nfeat={config.n_feature_maps}'
        
    experiment_name += f'-drprob={config.dqn_pos_reward_prob}'
    experiment_name += f'-dep={config.dqn_model_updates}'
    experiment_name += f'-dc={config.checkpoint_dqn}'
    experiment_name += f'-dev={config.dqn_eval_ep}'        
    experiment_name += f'-dlrt={config.dqn_learning_rate}'
    experiment_name += f'-seed={config.seed}'

    return config, experiment_name
        
    
