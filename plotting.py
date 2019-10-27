import os, pdb, torch, torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_plots(config, experiment, epoch, stats, split, pretrain=True, eps=None, goal='explore'):

    
    # folder to save visualizations
    viz_dir = f'{experiment}/viz/ep{epoch:05d}/{split}/{goal}'
    if not os.path.isdir(viz_dir):
        os.makedirs(viz_dir)
    

    if pretrain:
        '''
        # plot uncertainty across time
        plt.plot(range(u_list.size(0)), u_list.mean(1).numpy(), '.')
        plt.xlabel('rollout')
        plt.ylabel('mean uncertainty')
        plt.savefig(f'{experiment}/u_vs_time_ep{epoch}.pdf')
        plt.close()
        u_list_ = u_list.view(-1).numpy()
        mse_list_ = mse_list.view(-1).numpy()
        u_expert_ = u_expert.cpu().numpy()
        mse_expert_ = mse_expert.cpu().numpy()
        '''


        # plot uncertainty vs. MSE
        plt.plot(stats['uncertainty'].view(-1).numpy(), stats['mse'].view(-1).numpy(), '.', markersize=1, color='black')
        plt.xlabel('uncertainty')
        plt.ylabel('MSE')
        plt.savefig(f'{experiment}/u_vs_mse_ep{epoch}.pdf')

    # plot some trajectories in the replay buffer
    print('[saving trajectory viz]')
    ep_dir = viz_dir + '/experience/'

    os.system('mkdir -p ' + ep_dir)
    n_episodes = stats['uncertainty'].size(0)
    if config.input_type == 'image':
        # latest ones
        ep_list = [i for i in range(0, 5)]
    else:
        # a few older trajectories
        ep_list = [i for i in range(0, n_episodes, int(config.n_exploration_episodes/2))]
        ep_list += [i for i in range(0, 5)]
    ep_list = list(set(ep_list))

    # get top MSE quantile, so all plots are comparable scale
    mse_top_quant = torch.sort(stats['mse'].view(-1), descending=False)[0][round(config.u_quantile*stats['mse'].numel())]

    for j in ep_list:
        s_real = stats['s_real'][j]
        s_pred = stats['s_pred'][j]
        actions = stats['actions'][j]
        u = stats['uncertainty'][j]
        mse = stats['mse'][j]
        search_graph = stats['search_graph'][j]
        
        timestep = torch.arange(s_pred.size(0))
        if s_pred.dim() == 3:
            timestep_pred = timestep.unsqueeze(1).repeat(1, s_pred.size(1)).view(-1)
            s_pred_mean = s_pred.mean(1)
            s_pred = s_pred.view(s_pred.size(0)*s_pred.size(1), -1)
        else:
            timestep_pred = timestep
            s_pred_mean = s_pred
        
        if eps is None:
            eps = stats['eps']
        u = u / eps # plotting doesn't like small numbers so normalize
                  
        if config.env == 'mountaincar' or config.env == 'acrobot':
            plt.close()
            fig = plt.figure()
            dotsize=2
            ax = plt.subplot(3, 2, 1)
            ax.set_title('Truth (z=time)')
            ax.scatter(s_real[:, 0].numpy(), s_real[:, 1].numpy(), s=dotsize, c=timestep.numpy())
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            ax = plt.subplot(3, 2, 2)
            ax.set_title('Pred (z=time)')
            ax.scatter(s_pred[:, 0].numpy(), s_pred[:, 1].numpy(), s=dotsize, c=timestep_pred.numpy())
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            ax = plt.subplot(3, 2, 3)            
            ax.set_title('Truth (z=uncertainty)')
            ax.scatter(s_real[:, 0].numpy(), s_real[:, 1].numpy(), s=dotsize, c=u.numpy(), vmin=0, vmax=1)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            ax = plt.subplot(3, 2, 4)
            ax.set_title('Pred (z=uncertainty)')
            ax.scatter(s_pred_mean[:, 0].numpy(), s_pred_mean[:, 1].numpy(), s=dotsize, c=u.numpy(), vmin=0, vmax=1)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            ax = plt.subplot(3, 2, 5)
            ax.set_title('Truth (z=MSE)')
            ax.scatter(s_real[:, 0].numpy(), s_real[:, 1].numpy(), s=dotsize, c=mse.numpy(), vmin=0, vmax=mse_top_quant)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            ax = plt.subplot(3, 2, 4)
            ax.set_title('Pred (z=MSE)')
            ax.scatter(s_pred_mean[:, 0].numpy(), s_pred_mean[:, 1].numpy(), s=dotsize, c=mse.numpy(), vmin=0, vmax=mse_top_quant)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            


            save_file = f'{ep_dir}/traj{j}'
            if pretrain:
                save_file += '_pretrain.pdf'
            else:
                save_file += '_posttrain.pdf'
            print(f'[saving to: {save_file}]')
            plt.savefig(save_file)
            plt.close()

            if search_graph is not None:
                if search_graph.dim() == 3:
                    search_graph = torch.mean(search_graph, 1)
                plt.scatter(search_graph[:, 0].cpu().numpy(), search_graph[:, 1].cpu().numpy(), s=1)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                save_file = f'{ep_dir}/search_graph{j}.pdf'
                print(f'[saving to: {save_file}]')
                plt.savefig(save_file)
                plt.close()
                
        elif config.input_type == 'image':
            if config.phi == 'learned' or config.phi == 'ae':
                pred = stats['s_pred'][:, :200].float() 
                if config.loss == 'softmax':
                    pred = pred / 255.0
                images = torch.cat((stats['s_real'][:, :200].unsqueeze(2), pred), 2)
            elif config.phi == 'random':
                images = stats['s_real']
            ep_dir = viz_dir + f'/ep{j}/'
            print(f'[saving movie {ep_dir}]')
            if not os.path.isdir(ep_dir):
                os.makedirs(ep_dir)
            for t in range(images.size(1)):
                torchvision.utils.save_image(images[j][t], f'{ep_dir}/step{t:03d}_a{actions[t]}.png', nrow=config.n_ensemble+1)
                                
    
    os.system(f'tar -cf {experiment}/viz_ep{epoch}.tar.gz {experiment}/viz')
