import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_episodes_loss(data_array, key, n_joints='', num_ep='', len_ep='',
                       res_levels='', x_label='', y_label='', title='',
                       l_width=2, save_name='', plot_dir=os.getcwd(),
                       want_one=False):

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir, 0o777)

    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    ax.grid(True)
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.set_title(title, fontsize=32)
    if isinstance(data_array[key], dict):
        avg = [data_array[key][x] for x in list(data_array[key].keys())]
        ax.plot(np.average(avg, axis=0), linewidth=l_width)
    else:
        ax.plot(data_array[key], linewidth=l_width)
    fig.tight_layout()
    fig.savefig(plot_dir + os.sep + save_name +
                '_{}J_{}E_{}EL_{}RES.png'
                .format(n_joints, num_ep, len_ep, res_levels), dpi=200)
    plt.close()


def plot_sum_dict(data_dict, x_label='', y_label='', title='', save_name='',
                  plot_dir=os.getcwd()):
    fig, ax = plt.subplots(dpi=200, figsize=(16, 9))
    ax.grid(True)
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.set_title(title, fontsize=32)

    labels = []
    for k, vals in list(data_dict.items()):
        ax.plot(vals, linewidth=4)
        labels.append(k)

    ax.legend(labels, fontsize=18, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, -0.5))
    fig.tight_layout(h_pad=0, pad=0, rect=(0, 0, 1, 1))
    fig.savefig(plot_dir + os.sep + save_name + '.png', dpi=200,
                transparent=True)
    plt.close()


if __name__ == '__main__':
    NUM_J = [1, 2]
    NUM_EP = [500, 1300]
    LEN_EP = [256]
    RES_LVL = [11, 17]

    plot_dir = os.getcwd() + os.sep + 'Figures'
    searchDir = os.getcwd() + os.sep + 'Collected_Data'

    summary_d = {'cost_last_ep_random_pos': {}, 'cost_last_ep_down_pos': {},
                 'cost_best_net_down_pos': {}, 'cost_best_net_random_pos': {}}
    summary = {}
    for nj in NUM_J:
        summary[nj] = {'cost_last_ep_random_pos': {},
                       'cost_last_ep_down_pos': {},
                       'cost_best_net_down_pos': {},
                       'cost_best_net_random_pos': {}}
        for ne in NUM_EP:
            for le in LEN_EP:
                for rl in RES_LVL:
                    descriptor = '{}_joints_{}_ep_{}_len_{}_res'\
                                 .format(nj, ne, le, rl)
                    nameFile = 'Results_' + descriptor + '.json'
                    file_in = searchDir + os.sep + nameFile

                    if not os.path.isfile(file_in):
                        print('No file {} found at {}'
                              .format(nameFile, searchDir))
                    io_in = open(file_in, 'r')
                    data = json.load(io_in)
                    io_in.close()

                    # for key in data.keys():
                    #     plot_episodes_loss(data, key, nj, ne, le, rl,
                    #                        save_name=key, plot_dir=plot_dir)

                    for key in list(summary[nj].keys()):
                        avg = np.average([data[key][x]
                                          for x in list(data[key].keys())],
                                         axis=0)
                        summary[nj][key][descriptor] = avg
    for nj in NUM_J:
        for key in list(summary[nj].keys()):
            plot_sum_dict(summary[nj][key], y_label='Cumulative Sum',
                          x_label='Episodes', title=key,
                          save_name='Summary_%s_%dJ' % (key, nj),
                          plot_dir=plot_dir)
