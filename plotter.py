import json
import os
import matplotlib.pyplot as plt


def plot_episodes_loss(data_array, key, n_joints, num_ep, len_ep, res_levels,
                       searchDir=os.getcwd(), x_label='', y_label='', title='',
                       l_width=2, save_name='', plot_dir=os.getcwd()):

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir, 0o777)

    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    ax.grid(True)
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.set_title(title, fontsize=32)
    if isinstance(data_array[key], dict):
        for k, vals in data_array[key].items():
            ax.plot(vals, linewidth=l_width)
    else:
        ax.plot(data_array[key], linewidth=l_width)
    fig.legend(range(1, 4))
    fig.tight_layout()
    fig.savefig(plot_dir + os.sep + save_name +
                          '_{}J_{}E_{}EL.png'
                          .format(n_joints, num_ep, len_ep), dpi=200)

    '''
    # Average loss over the episodes plot
    ep_average_loss, ep_av_l_ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    tr_time, time_ax = plt.subplots(1, dpi=200, figsize=(16, 9))

    ep_av_l_ax.grid(True)
    ep_av_l_ax.set_xlabel('Number of episodes', fontsize=24)
    ep_av_l_ax.set_ylabel('Average loss', fontsize=24)
    ep_av_l_ax.set_title('Average Loss over Episodes', fontsize=32)

    time_ax.grid(True)
    time_ax.set_xlabel('Number of episodes', fontsize=24)
    time_ax.set_ylabel('time[s]', fontsize=24)
    time_ax.set_title('Training time %d joints' % n_joints, fontsize=32)

    av_loss = []
    training_time = []
    for el in range(num_ep):
        av_loss.append(data[str(el)]['loss'])
        training_time.append(data[str(el)]['time'])
    ep_av_l_ax.plot(av_loss)
    time_ax.plot(training_time[1:])

    ep_average_loss.tight_layout()
    tr_time.tight_layout()
    ep_average_loss.savefig(plot_dir + os.sep +
                            'average_loss_over_epiodes_{}J_{}E_{}EL.png'
                            .format(n_joints, num_ep, len_ep), dpi=200)

    tr_time.savefig(plot_dir + os.sep +
                    'training_time_over_epiodes_{}J_{}E_{}EL.png'
                    .format(n_joints, num_ep, len_ep), dpi=200)

    # Loss last episode Random
    loss_last_ep_random, ller_ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    ller_ax.grid(True)
    ller_ax.set_xlabel('Number of episodes', fontsize=24)
    ller_ax.set_ylabel('Cumulative cost', fontsize=24)
    ller_ax.set_title('Cost last episode with random position', fontsize=32)
    for key, vals in data['loss_last_ep_random_pos'].items():
        ller_ax.plot(vals, linewidth=4)
    loss_last_ep_random.legend(range(1, 4))
    loss_last_ep_random.tight_layout()
    loss_last_ep_random.savefig(plot_dir + os.sep +
                                'loss_last_ep_random_positions_{}J_{}E_{}EL.png'
                                .format(n_joints, num_ep, len_ep), dpi=200)

    # Loss last episode Down
    loss_last_ep_down, lled_ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    lled_ax.grid(True)
    lled_ax.set_xlabel('Number of episodes', fontsize=24)
    lled_ax.set_ylabel('Cumulative cost', fontsize=24)
    lled_ax.set_title('Cost last episode with down position', fontsize=32)
    for key, vals in data['loss_last_ep_down_pos'].items():
        lled_ax.plot(vals, linewidth=4)
    loss_last_ep_down.legend(range(1, 4))
    loss_last_ep_down.tight_layout()
    loss_last_ep_down.savefig(plot_dir + os.sep +
                              'loss_last_ep_down_positions_{}J_{}E_{}EL.png'
                              .format(n_joints, num_ep, len_ep), dpi=200)

    # Loss best network random
    best_net_random, bnr_ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    bnr_ax.grid(True)
    bnr_ax.set_xlabel('Number of episodes', fontsize=24)
    bnr_ax.set_ylabel('Cumulative cost', fontsize=24)
    bnr_ax.set_title('Cost best network with random position', fontsize=32)
    for key, vals in data['loss_best_net_random_pos'].items():
        bnr_ax.plot(vals, linewidth=4)
    best_net_random.legend(range(1, 4))
    best_net_random.tight_layout()
    best_net_random.savefig(plot_dir + os.sep +
                            'loss_best_net_random_positions_{}J_{}E_{}EL.png'
                            .format(n_joints, num_ep, len_ep), dpi=200)

    # Loss best network random
    best_net_down, bnd_ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    bnd_ax.grid(True)
    bnd_ax.set_xlabel('Number of episodes', fontsize=24)
    bnd_ax.set_ylabel('Cumulative cost', fontsize=24)
    bnd_ax.set_title('Cost best network with down position', fontsize=32)
    for key, vals in data['loss_best_net_down_pos'].items():
        bnd_ax.plot(vals, linewidth=4)
    best_net_down.legend(range(1, 4))
    best_net_down.tight_layout()
    best_net_down.savefig(plot_dir + os.sep +
                          'loss_best_net_down_positions_{}J_{}E_{}EL.png'
                          .format(n_joints, num_ep, len_ep), dpi=200)
    '''


if __name__ == '__main__':
    NUM_J = [2]
    NUM_EP = [500, 1000]
    LEN_EP = [256]
    RES_LVL = [15]

    plot_dir = os.getcwd() + os.sep + 'Figures'
    searchDir = os.getcwd() + os.sep + 'Collected_Data'

    for nj in NUM_J:
        for ne in NUM_EP:
            for le in LEN_EP:
                for rl in RES_LVL:
                    nameFile = 'Results_{}_joints_{}_ep_{}_len_{}_res.json'\
                               .format(nj, ne, le, rl)
                    file_in = searchDir + os.sep + nameFile

                    if not os.path.isfile(file_in):
                        print('No file {} found at {}'
                              .format(nameFile, searchDir))
                    io_in = open(file_in, 'r')
                    data = json.load(io_in)
                    io_in.close()

                    for key in data.keys():
                        plot_episodes_loss(data, key, nj, ne, le, rl,
                                           save_name=key, plot_dir=plot_dir)
