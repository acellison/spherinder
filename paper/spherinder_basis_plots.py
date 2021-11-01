import os

from spherinder.operators import Basis
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


def output_filename(directory, ext, prefix='modes'):
    return make_filename_prefix(directory) + f'-{prefix}' + ext


def save_figure(filename, fig, *args, **kwargs):
    checkdir(filename)
    fig.savefig(filename, *args, **kwargs)


def plotfield_full(m, s, eta, f, fig=None, ax=None, aspect='equal', cmap='RdBu', fontsize=12):
    """Plot a 2D slice of the field at phi = 0"""
    ss, ee = s.ravel()[np.newaxis,:], eta.ravel()[:,np.newaxis]
    left_rect = False
    if left_rect:
        yl, yr = 0*ss+ee, np.sqrt(1-ss**2)*ee
    else:
        yl, yr = np.sqrt(1-ss**2)*ee, 0*ss+ee
    yy = np.hstack([yl[:,::-1], yr[:,1:]])
    ss = np.hstack([-ss[:,::-1], ss[:,1:]])
    sign = np.exp(1j*m*np.pi).real
    ff = np.hstack([sign*f[:,::-1], f[:,1:]])

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4.25,6))

    im = ax.pcolormesh(ss, yy, ff, cmap=cmap, shading='auto')

    def plot_line(x, y):
        eps = 0.0125
        ax.plot(x, (1+eps)*np.array(y), 'k', linewidth=0.5)

    xbox = np.linspace(-1,1, 2*len(s)+1)
    if left_rect:
        pred, xv = (xbox > 0), -1
    else:
        pred, xv = (xbox < 0), 1
    ybox = np.where(pred, np.sqrt(1-xbox**2), 1.)
    plot_line(xbox,  ybox)
    plot_line(xbox, -ybox)
    plot_line((xv,)*2, [-1, 1])

    if aspect is not None:
        ax.set_aspect(aspect, adjustable='datalim')

    ax.set_xlabel('x', fontsize=fontsize)
    ax.set_xticks(np.linspace(-1,1,5))

    ax.set_ylabel('z', fontsize=fontsize)
    ax.set_yticks(np.linspace(-1,1,5))
    ax.yaxis.set_ticks_position('both')

    fig.set_tight_layout(True)
    return fig, ax


def plot_basis(fig, plot_axes, basis, m, ell, s, eta):
    fontsize = 12
    for k in range(basis.Nmax):
        ax, field = plot_axes[k], basis[ell,k]
        plotfield_full(m, s, eta, field, fig, ax, fontsize=fontsize)

        ax.plot([0,0],[-1,1],'--k', linewidth=1, alpha=0.5)
        ax.set_xticks(np.linspace(-1,1,5))
        if k == 0:
            ax.yaxis.set_label_position('left')
        elif k == basis.Nmax-1:
            ax.yaxis.tick_right()
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_label_position('right')
            ax.set_ylabel('η', fontsize=fontsize)
        else:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        ax.text(-1.0,-1.0, f'({m}, {ell}, {k})', fontsize=fontsize)

def main():
    ns, neta = 1024, 1025
    s, eta = np.linspace(0,1,ns), np.linspace(-1,1,neta)

    Nmax = 4
    configs = [(0,0), (1,8), (8,1)]
    figscale = 1.5

    nrows, ncols = len(configs), Nmax
    delta = (1.,0.2)
    figsize = tuple(figscale*a+d for (a,d) in zip(plt.figaspect(nrows/ncols), delta))
    fig, plot_axes = plt.subplots(nrows,ncols,figsize=figsize)

    for i, (m,ell) in enumerate(configs):
        basis = Basis(s, eta, m, ell+1, Nmax, sigma=0, alpha=-1/2, truncate=False)
        plot_basis(fig, plot_axes[i], basis, m, ell, s, eta)
        if i < nrows-1:
            for ax in plot_axes[i]:
                ax.set_xlabel('')
                ax.set_xticklabels([])

    fig.set_tight_layout(True)

    ext = '-full.png'
    filename = output_filename(directory='figures', ext=ext, prefix='modes')
    save_figure(filename, fig, dpi=200)


if __name__=='__main__':
    main()
    plt.show()

