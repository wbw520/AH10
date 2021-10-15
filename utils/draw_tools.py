import matplotlib.pyplot as plt
import numpy as np


cls_color = ['#000080',  # navy
            '#FF0000',  # red
            '#FFFF00',  # yellow
            '#FF8C00',  # darkorange
            '#800080',  # purple
            '#008080',  # teal
            '#008000',  # green
            '#000000',  # black
            '#C0C0C0',  # silver
            '#A52A2A',  # brown
            '#8B0000',  # darkred
            ]


def plot_embedding(data, label, cat, ax):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    coor_list_x = [[] for i in range(len(cat))]
    coor_list_y = [[] for i in range(len(cat))]
    for i in range(len(data)):
        id = int(label[i])
        coor_list_x[id].append(data[i][0])
        coor_list_y[id].append(data[i][1])

    for j in range(len(cat)):
        color = cls_color[cat.index(cat[j])]
        ax.scatter(coor_list_x[j], coor_list_y[j], color=color, s=60, label=cat[j])

    ax.set_xticks([])
    ax.set_yticks([])


def draw_umap(data, label, cls_name):
    plt.figure(figsize=(16, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 3
    font = 31

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.set_title("w/o Knowledge Transfer", fontsize=font)
    ax2.set_title("w/ Knowledge Transfer", fontsize=font)
    plot_embedding(data[0], label[0], cls_name, ax1)
    plot_embedding(data[1], label[1], cls_name, ax2)

    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=25, labelspacing=0.5, markerscale=1.9,
               handletextpad=0.1)
    plt.tight_layout()
    plt.savefig("umap.pdf")
    plt.show()