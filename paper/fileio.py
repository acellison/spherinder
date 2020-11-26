import os
import pickle

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


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
    
