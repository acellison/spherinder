import numpy as np
import matplotlib.pyplot as plt

import spherinder.operators as sph


m, Lmax, Nmax, alpha, sigma = 1, 8, 10, 0, 0
ell, k = 4, 2

ncoeff = Lmax*Nmax
markersize = 800
center = 's'
markers = ['d','X','P']
margin = .2


# Make sure we're looking at the densest column
def max_density(op):
    maxnz, maxcol = 0, 0
    for c in range(np.shape(op)[1]):
        nzr, _ = op[:,c].nonzero()
        if len(nzr) > maxnz:
            maxnz = len(nzr)
            maxcol = c
    return maxnz, maxcol


def plot_splatter(opname, operator, codomain, ax=None, margins=(margin,margin)):
    if not isinstance(operator, (list,tuple)):
        operator = [operator]
        codomain = [codomain]

    ellmin, ellmax, kmin, kmax = np.inf, -np.inf, np.inf, -np.inf
    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3))
        return_fig = True
    else:
        return_fig = False

    ax.scatter(0,0,marker=center,color='tab:purple',s=markersize)
    for i, op in enumerate(operator):
        ells, ks = operator_splatter(op, codomain[i][1], Nmax, ell, k)
        ax.scatter(ks,ells,s=markersize,marker=markers[i])

        ellmin, ellmax = min(min(ells), ellmin), max(max(ells), ellmax)
        kmin, kmax = min(min(ks), kmin), max(max(ks), kmax)

    ax.set_xticks(range(int(kmin),int(kmax)+1))
    ax.set_yticks(range(int(ellmin),int(ellmax)+1))
    ax.set_aspect('equal')
    ax.margins(*margins)
    ax.set_xlabel('$k$')
    ax.set_ylabel('$l$')
    ax.set_title(opname)

    if return_fig:
        return fig, ax


def operator_splatter(op, Nout, Nin, ell, k):
    colindex = ell*Nin + k
    col = op[:,colindex]
    rows, _ = col.nonzero()
    maxnz, maxcol = max_density(op)
    if len(rows) != maxnz:
        raise ValueError(f'Not your densest column, {maxcol} is denser')

    inds = [(r//Nout,r%Nout) for r in rows]
    ells, ks = zip(*inds)
    ells, ks = np.asarray(ells)-ell, np.asarray(ks)-k
    return ells, ks


def differential_operators():
    # Differential operators
    fig, ax = plt.subplots(1,3,figsize=(9,3))
    
    # Gradient sparsity structure
    operator = sph.operator('gradient')
    codomain = [cd(Lmax,Nmax,alpha) for cd in operator.codomain]
    Op = operator(m,Lmax,Nmax,alpha)
    plot_splatter('Gradient', Op, codomain, ax=ax[0])
    
    # Divergence sparsity structure
    operator = sph.operator('divergence')
    codomain = [(Lmax,Nmax+1)]*3
    Op = operator(m,Lmax,Nmax,alpha)
    ops = [Op[:,:ncoeff],Op[:,ncoeff:2*ncoeff],Op[:,2*ncoeff:3*ncoeff]]
    plot_splatter('Divergence', ops, codomain, ax=ax[1])
    
    # Laplacian sparsity structure
    operator = sph.operator('laplacian')
    codomain = operator.codomain(Lmax,Nmax,alpha)
    Op = operator(m,Lmax,Nmax,alpha)
    plot_splatter('Laplacian', Op, codomain, ax=ax[2], margins=(1.5*margin,margin))

    fig.set_tight_layout(True)


def radial_operators():
    # radial operators
    fig, ax = plt.subplots(1,4,figsize=(12,3))
    
    # r e_r sparsity structure
    operator = sph.RadialMultiplication(convert=False)
    codomain = [cd(Lmax,Nmax,alpha) for cd in operator.codomain]
    Op = operator(m,Lmax,Nmax,alpha)
    plot_splatter('$r$ $e_r$', Op, codomain, ax=ax[0])
    
    # e_r . sparsity structure
    operator = sph.operator('erdot')
    codomain = [(Lmax+1,Nmax+1)]*3
    Op = operator(m,Lmax,Nmax,alpha)
    ops = [Op[:,:ncoeff],Op[:,ncoeff:2*ncoeff],Op[:,2*ncoeff:3*ncoeff]]
    plot_splatter('$e_r \cdot$', ops, codomain, ax=ax[1])
    
    # 1-r**2 sparsity structure
    operator = sph.operator('1-r**2')
    codomain = operator.codomain(Lmax,Nmax,alpha)
    Op = operator(m,Lmax,Nmax,alpha=1,sigma=0)
    plot_splatter('$1-r^2$', Op, codomain, ax=ax[2])
    
    # conversion sparsity structure
    operator = sph.operator('conversion')
    codomain = operator.codomain(Lmax,Nmax,alpha)
    Op = operator(m,Lmax,Nmax,alpha=0,sigma=0)
    plot_splatter('$\\alpha$-conversion', Op, codomain, ax=ax[3])

    fig.set_tight_layout(True)


def main():
    differential_operators()
    radial_operators()
    plt.show()

if __name__=='__main__':
    main()

