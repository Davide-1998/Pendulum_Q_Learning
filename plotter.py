import json
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_episodes_loss(num_ep, len_ep, searchDir=os.getcwd()):
    plot_dir = os.getcwd() + os.sep + 'Figures'
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir, 0o777)

    nameFile = 'Results_2_joints_%d_ep_%d_len.json' % (num_ep, len_ep)
    file_in = searchDir + os.sep + nameFile

    if not os.path.isfile(file_in):
        print('No file {} found at {}'.format(nameFile, searchDir))
        return

    io_in = open(file_in, 'r')
    data = json.load(io_in)
    io_in.close()

    # Average loss over the episodes plot
    ep_average_loss, ep_av_l_ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    ep_av_l_ax.grid(True)
    ep_av_l_ax.set_xlabel('Number of episodes', fontsize=18)
    ep_av_l_ax.set_ylabel('Average loss', fontsize=18)
    ep_av_l_ax.set_title('Average Loss over Episodes', fontsize=24)

    av_loss = []
    for el in range(num_ep):
        av_loss.append(data[str(el)]['loss'])
    ep_av_l_ax.plot(av_loss)
    ep_average_loss.savefig(plot_dir + os.sep +
                            'average_loss_over_epiodes.png', dpi=200)

    # Loss last episode Random
    loss_last_ep_random, ller_ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    ller_ax.grid(True)
    ller_ax.set_xlabel('Number of episodes', fontsize=18)
    ller_ax.set_ylabel('Loss', fontsize=18)
    ller_ax.set_title('Loss last episode with random position', fontsize=24)
    for key, vals in data['loss_last_ep_random_pos'].items():
        ller_ax.plot(vals)
    loss_last_ep_random.legend(range(1, 4))
    loss_last_ep_random.savefig(plot_dir + os.sep +
                                'loss_last_ep_random_positions.png', dpi=200)

    # Loss last episode Down
    loss_last_ep_down, lled_ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    lled_ax.grid(True)
    lled_ax.set_xlabel('Number of episodes', fontsize=18)
    lled_ax.set_ylabel('Loss', fontsize=18)
    lled_ax.set_title('Loss last episode with down position', fontsize=24)
    for key, vals in data['loss_last_ep_down_pos'].items():
        lled_ax.plot(vals)
    loss_last_ep_down.legend(range(1, 4))
    loss_last_ep_down.savefig(plot_dir + os.sep +
                              'loss_last_ep_down_positions.png', dpi=200)

    # Loss best network random
    best_net_random, bnr_ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    bnr_ax.grid(True)
    bnr_ax.set_xlabel('Number of episodes', fontsize=18)
    bnr_ax.set_ylabel('Loss', fontsize=18)
    bnr_ax.set_title('Loss best network with random position', fontsize=24)
    for key, vals in data['loss_best_net_random_pos'].items():
        bnr_ax.plot(vals)
    best_net_random.legend(range(1, 4))
    best_net_random.savefig(plot_dir + os.sep +
                            'loss_best_net_random_positions.png', dpi=200)

    # Loss best network random
    best_net_down, bnd_ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    bnd_ax.grid(True)
    bnd_ax.set_xlabel('Number of episodes', fontsize=18)
    bnd_ax.set_ylabel('Loss', fontsize=18)
    bnd_ax.set_title('Loss best network with down position', fontsize=24)
    for key, vals in data['loss_best_net_down_pos'].items():
        bnd_ax.plot(vals)
    best_net_down.legend(range(1, 4))
    best_net_down.savefig(plot_dir + os.sep +
                          'loss_best_net_down_positions.png', dpi=200)


if __name__ == '__main__':
    NUM_EP = 500
    LEN_EP = 256

    plot_episodes_loss(NUM_EP, LEN_EP)
