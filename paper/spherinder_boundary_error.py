import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import scipy.linalg as spla
from mpmath import mp

import os
import pickle

import spherinder.operators as sph


def null_space(A, rcond=None, kind='scipy'):
    if kind == 'scipy':
        svd = spla.svd
        nrows = lambda x: x.shape[0]
        ncols = lambda x: x.shape[1]
    elif kind == 'mpmath':
        svd = mp.svd
        A = mp.matrix(A)
        nrows = lambda x: x.rows
        ncols = lambda x: x.cols

    u, s, vh = svd(A, full_matrices=True)
    if rcond is None:
        M, N = nrows(u), ncols(vh)
        rcond = np.finfo(s.dtype).eps * max(M, N)
    if rcond > 0:
        tol = np.amax(s) * rcond
        num = np.sum(s > tol, dtype=int)
    else:
        num = len(s)

    if kind == 'mpmath':
        vh = np.array(vh.tolist(), dtype='float128')
    Q = vh[num:,:].T.conj()
    return Q


def analyze(m, Lmax, Nmax, alpha, sigma, dtype='float64'):
    mat = sph.operator('boundary', dtype=dtype, internal=dtype)(m, Lmax, Nmax, alpha=alpha, sigma=sigma)
    nullspace1 = spla.null_space(mat.todense())

    if True:
        mp.dps = 50
        kind = 'mpmath'
    else:
        kind = 'scipy'
    nullspace = null_space(mat.todense(), rcond=0., kind=kind)

    ns = 64
    s, eta = np.linspace(0,1,ns,dtype=dtype), np.array([-1.,1.], dtype=dtype)
    basis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=sigma, alpha=alpha, dtype=dtype)

    nvecs = np.shape(nullspace)[1]
    solns = [basis.expand(nullspace[:,i]) for i in range(nvecs)]
    errors = [np.max(abs(solns[i])) for i in range(nvecs)]
    index = np.argmax(errors)
    coeffs = nullspace[:,index]

    boundary = mat @ coeffs
    
    grid_error = max(errors)
    coeff_error_2 = np.linalg.norm(boundary)
    coeff_error_max = np.max(abs(boundary))
    print(f'm = {m}, Lmax = {Lmax}, Nmax = {Nmax}, grid error: {grid_error}, coeff error: {coeff_error_2} (L2), {coeff_error_max} (max)')

    for ell in range(0,min(11,Lmax),2):
        print(f'ell = {ell}: ', coeffs[ell*Nmax:(ell+1)*Nmax])

    fig, ax = plt.subplots()
    ax.semilogy(s, abs(solns[index][0,:]))
    ax.semilogy(s, abs(solns[index][1,:]))

    # Plot the most-offending field
    ns, neta = 250, 201
    s, eta = np.linspace(0,1,ns), np.linspace(-1,1,neta)
    basis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=sigma, alpha=alpha)
    f = basis.expand(coeffs)

    fig, ax = plt.subplots()
    sph.plotfield(s, eta, f, fig=fig, ax=ax, stretch=False, aspect='equal')

    plt.show()


def main():
#    m, Lmax, Nmax = 0, 20, 80
#    m, Lmax, Nmax = 0, 29, 29  # First square suck
    m, Lmax, Nmax = 0, 31, 20
    alpha, sigma = 0, 0
    dtype = 'float128'

    analyze(m, Lmax, Nmax, alpha, sigma, dtype=dtype)
    


if __name__=='__main__':
    main()
