import numpy as np

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'

import matplotlib.pyplot as plt
import os

import spherinder.operators as sph
from fileio import save_figure


m, Lmax, Nmax, alpha, sigma = 1, 16, 20, 0, 0
ell, k = 8, 4

ncoeff = sph.num_coeffs(Lmax, Nmax)
markersize = 800
center = 's'
markers = ['P','_','o']
linewidths = [None, 8, None]
margin = .2

markerdict = {'+': {'color': 'tab:green',  'marker': 'P', 'linewidth': None},
              '-': {'color': 'tab:orange', 'marker': '_', 'linewidth': 8},
              '0': {'color': 'tab:blue',   'marker': 'o', 'linewidth': None}}


g_file_prefix = 'spherinder_sparsity'


# Make sure we're looking at the densest column
def max_density(op):
    maxnz, maxcol = 0, 0
    for c in range(np.shape(op)[1]):
        nzr, _ = op[:,c].nonzero()
        if len(nzr) > maxnz:
            maxnz, maxcol = len(nzr), c
    return maxnz, maxcol


def plot_splatter(opname, operator, codomain, ax=None, margins=(margin,margin), flip=False, aspect='equal'):
    if not isinstance(operator, (list,tuple)):
        order = ['0']
        operator = [operator]
        codomain = [codomain]
        plotorder = [0]
    else:
        if flip:
            order = ['-','+','0']
            plotorder = [2, 1, 0]  # Keep '-' on top
        else:
            order = ['+','-','0']
            plotorder = [2, 0, 1]

    ellmin, ellmax, kmin, kmax = np.inf, -np.inf, np.inf, -np.inf
    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3))
        return_fig = True
    else:
        return_fig = False

    ax.scatter(0,0,marker=center,color='tab:purple',s=markersize)
    for i in range(len(operator)):
        # Get the operator
        index = plotorder[i]
        op = operator[index]

        # Get the plot parameters
        params = markerdict[order[index]]
        color, marker, linewidth = params['color'], params['marker'], params['linewidth']

        # Compute the splatter then plot
        ells, ks = operator_splatter(op, codomain[index][0], codomain[index][1], Lmax, Nmax, ell, k)
        ax.scatter(ks, ells, s=markersize, marker=marker, linewidth=linewidth, color=color)

        # Keep track of ell and k extrema
        ellmin, ellmax = min(min(ells), ellmin), max(max(ells), ellmax)
        kmin, kmax = min(min(ks), kmin), max(max(ks), kmax)

    ellmax = max(ellmax, 0)
    if ellmin == ellmax:
        ax.set_ylim([1,-1])

    ax.set_xticks(range(int(kmin),int(kmax)+1))
    ax.set_yticks(range(int(ellmin),int(ellmax)+1))
    ax.set_aspect(aspect)
    ax.margins(*margins)
    ax.set_xlabel(r'$Δ k$')
    ax.set_ylabel(r'$Δ l$')
    ax.set_title(opname)

    if return_fig:
        return fig, ax


def get_ell_k(index, Lmax, Nmax):
    _, offsets = sph.coeff_sizes(Lmax, Nmax)
    ell = Lmax-1 - np.argmin(index < offsets[::-1])
    return (ell, index-offsets[ell])


def operator_splatter(op, Lout, Nout, Lin, Nin, ell, k):
    lengths, offsets = sph.coeff_sizes(Lin, Nin)
    colindex = offsets[ell] + k
    col = op[:,colindex]
    rows, _ = col.nonzero()
    maxnz, maxcol = max_density(op)
    if len(rows) != maxnz:
        raise ValueError(f'Not your densest column, {maxcol} is denser')

    inds = [get_ell_k(r, Lout, Nout) for r in rows]
    ells, ks = zip(*inds)
    ells, ks = np.asarray(ells)-ell, np.asarray(ks)-k
    return ells, ks


def make_filename_prefix(directory='data'):
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), directory))
    abspath = os.path.join(basepath, g_file_prefix)
    return os.path.join(abspath, g_file_prefix)


def output_filename(directory, ext, prefix):
    return make_filename_prefix(directory) + f'-{prefix}' + ext


def differential_operators():
    # Differential operators
    fig, ax = plt.subplots(1,3,figsize=(9,3))
    
    # Gradient sparsity structure
    operator = sph.operator('gradient')
    codomain = [cd(Lmax,Nmax,alpha) for cd in operator.codomain]
    Op = operator(m,Lmax,Nmax,alpha)
    plot_splatter(r'$\mathcal{G}$   (Gradient)', Op, codomain, ax=ax[0])
    
    # Divergence sparsity structure
    operator = sph.operator('divergence')
    codomain = [cd(Lmax,Nmax,alpha) for cd in operator.codomain]
    Op = operator(m,Lmax,Nmax,alpha)
    ops = [Op[:,:ncoeff],Op[:,ncoeff:2*ncoeff],Op[:,2*ncoeff:3*ncoeff]]
    plot_splatter(r'$\mathcal{D}$   (Divergence)', ops, codomain, ax=ax[1], flip=True)
    
    # Laplacian sparsity structure
    operator = sph.operator('laplacian')
    codomain = operator.codomain(Lmax,Nmax,alpha)
    Op = operator(m,Lmax,Nmax,alpha)
    plot_splatter(r'$\mathcal{L}$   (Laplacian)', Op, codomain, ax=ax[2], aspect=0.5)

    filename = output_filename('figures', ext='.png', prefix='differential_ops')
    save_figure(filename, fig)


def differential_operators_poster():
    # Differential operators
    figsize = plt.figaspect(0.5)
    figsize = [0.8*a for a in figsize]
    fig, ax = plt.subplots(1,2,figsize=figsize)

    # Laplacian sparsity structure
    operator = sph.operator('laplacian')
    codomain = operator.codomain(Lmax,Nmax,alpha)
    Op = operator(m,Lmax,Nmax,alpha)
    plot_splatter(r'$\mathcal{L}$   (Laplacian)', Op, codomain, ax=ax[0], aspect=0.5)

    # Gradient sparsity structure
    operator = sph.operator('gradient')
    codomain = [cd(Lmax,Nmax,alpha) for cd in operator.codomain]
    Op = operator(m,Lmax,Nmax,alpha)
    plot_splatter(r'$\mathcal{G}$   (Gradient)', Op, codomain, ax=ax[1])

    fig.set_tight_layout(True)
    filename = output_filename('figures', ext='.png', prefix='differential_ops-poster')
    save_figure(filename, fig)


def radial_operators():
    # radial operators
    fig, ax = plt.subplots(1,3,figsize=(9,3))
    
    # r e_r sparsity structure
    operator = sph.RadialMultiplication()
    codomain = [cd(Lmax,Nmax,alpha) for cd in operator.codomain]
    Op = operator(m,Lmax,Nmax,alpha,exact=True)
    plot_splatter(r'$\mathcal{R}$   $(r e_r)$', Op, codomain, ax=ax[0])
    
    # e_r . sparsity structure
    operator = sph.operator('rdot')
    codomain = [(Lmax+1,Nmax+1)]*3
    Op = operator(m,Lmax,Nmax,alpha,exact=True)
    ops = [Op[:,:ncoeff],Op[:,ncoeff:2*ncoeff],Op[:,2*ncoeff:3*ncoeff]]
    plot_splatter(r'$\mathcal{E}$   $(r e_r \cdot)$', ops, codomain, ax=ax[1], flip=True)
    
    # 1-r**2 sparsity structure
    operator = sph.operator('1-r**2')
    codomain = operator.codomain(Lmax,Nmax,alpha)
    Op = operator(m,Lmax,Nmax,alpha=1,sigma=0,exact=True)
    plot_splatter(r'$\mathcal{S}$   $\left(1-r^2\right)$', Op, codomain, ax=ax[2])

    filename = output_filename('figures', ext='.png', prefix='radial_ops')
    save_figure(filename, fig)


def conversion_operators():
    # conversion operators
    fig, ax = plt.subplots(1,1,figsize=(3,3))

    # conversion sparsity structure
    operator = sph.operator('conversion')
    codomain = operator.codomain(Lmax,Nmax,alpha)
    Op = operator(m,Lmax,Nmax,alpha=0,sigma=0)
    plot_splatter(r'$\mathcal{I}_{\alpha}$', Op, codomain, ax=ax)

    filename = output_filename('figures', ext='.png', prefix='conversion_ops')
    save_figure(filename, fig)


def boundary_operator():
    m, Lmax, Nmax, alpha, sigma = 1, 10, 8, 0, 0
    boundary = sph.Boundary()(m, Lmax, Nmax, alpha, sigma)
    fig, ax = plt.subplots(figsize=plt.figaspect(0.3))
    fig.set_tight_layout(True)
    ax.spy(boundary)

    filename = output_filename('figures', ext='.png', prefix='boundary')
    save_figure(filename, fig)


def main():
    differential_operators()
#    differential_operators_poster()
    radial_operators()
    conversion_operators()
    boundary_operator()
    plt.show()

if __name__=='__main__':
    main()

