import json
import os
import matplotlib.pyplot as plt
import numpy as np
import subprocess


def plot_episodes_loss(data_array, key, n_joints='', num_ep='', len_ep='',
                       res_levels='', x_label='', y_label='', title='',
                       l_width=2, save_name='', plot_dir=os.getcwd(),
                       want_one=False):

    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir, 0o777)

    fig, ax = plt.subplots(1, dpi=200, figsize=(16, 9))
    plt.tick_params(labelsize=20)
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


def plot_dict(data_dict, x_label='', y_label='', title='', save_name='',
              plot_dir=os.getcwd(), label=True, last_one=False,
              legend_title=''):
    fig, ax = plt.subplots(dpi=200, figsize=(14, 9))
    plt.tick_params(labelsize=20)
    ax.grid(True)
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.set_title(title, fontsize=32)

    labels = []
    if last_one:
        last_vals = []
        for k, list_of_vals in list(data_dict.items()):
            last_vals.append(list_of_vals[-1])
        ax.plot(last_vals, linewidth=2)
    else:
        for k, vals in list(data_dict.items()):
            ax.plot(vals, linewidth=4)
            labels.append(k)

    if label and len(labels) != 0:
        ax.legend(labels, fontsize=20, loc='lower center', ncol=2,
                  bbox_to_anchor=(0.5, -0.5), title=legend_title,
                  title_fontsize=20)
    fig.tight_layout(h_pad=0, pad=0)
    fig.savefig(plot_dir + os.sep + save_name + '.png', dpi=200,
                transparent=True)
    plt.close()


def make_video(frame_folder=os.getcwd()):
    for desc_folder in os.listdir(frame_folder):
        videos = {}
        for frame in os.listdir(frame_folder + os.sep + desc_folder):
            frame = frame.replace('.png', '').split('_')

            frame_name = ''
            for el in frame[0:4]:
                frame_name += el + '_'

            frame_desc = ''
            for el in frame[4:11]:
                frame_desc += el + '_'

            frame_idx = frame[-1]

            n_frame_name = frame_name + frame_desc
            frame_path = os.path.abspath(frame_folder + os.sep + desc_folder +
                                         os.sep + n_frame_name + frame_idx)

            if n_frame_name not in videos:
                videos[n_frame_name] = [frame_path + '.png']
            else:
                videos[n_frame_name].append(frame_path + '.png')
        for key in list(videos.keys()):
            videos[key].sort()
            command = ['convert']
            for el in videos[key]:
                command.append(el)
            command.append('{}/{}/{}.mp4'
                           .format(frame_folder, desc_folder, key))
            subprocess.run(command)


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

                    keys_to_plot = ['0', str(int(ne/4)), str(int(ne/2)),
                                    str(int(ne-1))]
                    plot_dict(data['cost_to_go'],
                              x_label='Episodes_length',
                              y_label='Cumulative cost',
                              plot_dir=plot_dir,
                              last_one=True,
                              title='training_cost_' + descriptor,
                              save_name='Training_costs_' + descriptor,
                              legend_title='Episodes indices')

                    for key in data.keys():
                        plot_episodes_loss(data, key, nj, ne, le, rl,
                                           save_name=key, plot_dir=plot_dir,
                                           title=key+'_'+descriptor)

                    for key in list(summary[nj].keys()):
                        avg = np.average([data[key][x]
                                          for x in list(data[key].keys())],
                                         axis=0)
                        summary[nj][key][descriptor] = avg

    for nj in NUM_J:
        for key in list(summary[nj].keys()):
            plot_dict(summary[nj][key], y_label='Cumulative Sum',
                      x_label='Episodes', title=key,
                      save_name='Summary_%s_%dJ' % (key, nj),
                      plot_dir=plot_dir)

    # make_video('Frames')
