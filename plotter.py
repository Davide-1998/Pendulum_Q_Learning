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
    plt.close()


if __name__ == '__main__':
    NUM_J = [2]
    NUM_EP = [100, 500, 1000]
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
