import os

from spherinder.operators import Basis, plotfield
import numpy as np
import matplotlib.pyplot as plt

g_file_prefix = 'spherinder_basis'


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
    fig.savefig(filename, *args, **kwargs)
    

def make_filename_prefix(directory='data'):
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), directory))
    abspath = os.path.join(basepath, g_file_prefix)
    return os.path.join(abspath, g_file_prefix)


def output_filename(m, ell, directory, ext, prefix='modes'):
    return make_filename_prefix(directory) + f'-{prefix}-m={m}-ell={ell}' + ext


def save_figure(filename, fig, *args, **kwargs):
    checkdir(filename)
    fig.savefig(filename, *args, **kwargs)
    

def plot_basis(basis, ell, s, eta):
    Nmax = basis.Nmax
    ncols = 4
    nrows = (Nmax+ncols-1)//ncols
    fig, plot_axes = plt.subplots(nrows,ncols,figsize=(2*ncols-1.5,3*nrows))
    if nrows == 1:
        plot_axes = [plot_axes]

    row, col = 0, 0
    for k in range(Nmax):
        ax = plot_axes[row][col]
        field = basis[ell,k]
        plotfield(s, eta, field, fig, ax, colorbar=False)

        # Adjust the axes labels
        if col != 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        if row != nrows-1:
            ax.set_xlabel('')
            ax.set_xticklabels([])

        col += 1
        if col == ncols:
            col = 0
            row += 1
    fig.set_tight_layout(True)
    
    filename = output_filename(basis.m, ell, directory='figures', ext='.png', prefix='modes')
    save_figure(filename, fig, dpi=200)


def main():
    ns, neta = 200, 201
    s, eta = np.linspace(0,1,ns), np.linspace(-1,1,neta)

    mrange = [0]
    for m in mrange:
        Lmax, Nmax = 5, 4
        basis = Basis(s, eta, m, Lmax, Nmax, sigma=0, alpha=0)
        ell_range = range(Lmax)
        for ell in ell_range:
            plot_basis(basis, ell, s, eta)


if __name__=='__main__':
    main()
    plt.show()

