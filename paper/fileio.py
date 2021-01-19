import os
import pickle
import numpy as np

import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt


def checkdir(filename):
    path = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(path):
        os.makedirs(path)


def save_data(filename, data):
    checkdir(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def save_figure(filename, fig, *args, **kwargs):
    checkdir(filename)
    fig.set_tight_layout(True)
    fig.savefig(filename, *args, **kwargs)
    

def plotspy(L, M, markersize=2.):
    fig, plot_axes = plt.subplots(1,2,figsize=(8.5,4))
    plot_axes[0].spy(L, markersize=markersize)
    plot_axes[1].spy(M, markersize=markersize)
    plot_axes[0].set_xlabel('L')
    plot_axes[1].set_xlabel('M')
    nticks = 4
    for ax in plot_axes:
        xticks = ax.get_xticks()
        xticks = xticks[np.logical_and(xticks > 0, xticks < np.shape(L)[1])]
        xticks = xticks[::len(xticks)//nticks]
        ax.set_xticks(xticks)

        yticks = ax.get_yticks()
        yticks = yticks[np.logical_and(yticks > 0, yticks < np.shape(L)[0])]
        yticks = yticks[::len(yticks)//nticks]
        ax.set_yticks(yticks)

    plot_axes[1].set_yticklabels([])

    return fig, plot_axes

