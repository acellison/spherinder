import time
import numpy as np
from scipy.linalg import eig
from scipy import sparse
import dedalus.public as de
import dedalus.tools.sparse as dts

def eigsort(A, B, profile=False, overwrite=False):
    cutoff = np.inf

    if profile:
        print("Starting eigenvalue computation...")
        evals_start = time.time()

    vals, vecs = eig(A, b=B, overwrite_a=overwrite, overwrite_b=overwrite)
    bad = (np.abs(vals) > cutoff)
    vals[bad] = np.nan
    vecs = vecs[:,np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]

    i = np.argsort(vals.real)
    vals, vecs = vals[i], vecs[:, i]

    if profile:
        evals_end = time.time()
        print("Eigenvalues took {:g} sec".format(evals_end-evals_start))

    return vals, vecs


def scipy_sparse_eigs(A, B, N, target, matsolver=None, profile=False):
    """
    Perform targeted eigenmode search using the scipy/ARPACK sparse solver
    for the reformulated generalized eigenvalue problem

        A.x = λ B.x  ==>  (A - σB)^I B.x = (1/(λ-σ)) x

    for eigenvalues λ near the target σ.

    Parameters
    ----------
    A, B : scipy sparse matrices
        Sparse matrices for generalized eigenvalue problem
    N : int
        Number of eigenmodes to return
    target : complex
        Target σ for eigenvalue search
    matsolver : matrix solver class
        Class implementing solve method for solving sparse systems.

    Other keyword options passed to scipy.sparse.linalg.eigs.
    """
    if matsolver is None:
        matsolver = de.matsolvers.matsolvers['SuperLUNaturalSpsolve'.lower()]

    if profile:
        print("  Starting eigenvalue computation...")
        evals_start = time.time()

    vals, vecs = dts.scipy_sparse_eigs(A, B, N, target, matsolver)

    cutoff = np.inf
    bad = (np.abs(vals) > cutoff)
    vals[bad] = np.nan
    vecs = vecs[:,np.isfinite(vals)]
    vals = vals[np.isfinite(vals)]

    i = np.argsort(vals.real)
    vals, vecs = vals[i], vecs[:, i]

    if profile:
        evals_end = time.time()
        print("  Eigenvalues took {:g} sec".format(evals_end-evals_start))

    return vals, vecs


def track_eigenpair_homotopy(A, B, C, lam, v, alphak=None, tol=1e-14, maxiter=10):
    """Solve the perturbed eigenproblem (A + C) v = lambda B v
    """
    n = np.prod(np.shape(v))
    v = np.reshape(np.asmatrix(v), (n,1))

    # Normalize the initial guess
    v /= np.linalg.norm(v)

    if alphak is None:
        alphak = [1.]

    timeit = True
    if timeit:
        evals_start = time.time()

    for k in range(len(alphak)):
        print('  Homotopy, alpha_k = {}'.format(alphak[k]))
        alpha = alphak[k]
        for i in range(maxiter):
            M00 = A + alpha*C - lam * B
            M01 = - B @ v
            M10 = v.H
            M11 = np.zeros((1,1))

            r0 = (A + alpha*C - lam * B) @ v
            r1 = 0.5 * (v.H @ v - 1)

            M = sparse.bmat([[M00, M01], [M10, M11]], format='csr')
            r = sparse.bmat([[r0],[r1]]).todense()

            # Sparse solve
            delta = sparse.linalg.spsolve(M, -r)
            dv = np.reshape(delta[:len(delta)-1], np.shape(v))
            dlam = delta[-1]

            v += dv
            lam += dlam

            print('    alpha: {:1.4f},  iter: {:3d},  delta norm: {:1.6e}'.format(alpha, i, np.linalg.norm(delta)))
            if np.linalg.norm(delta) <= tol:
                break

    if timeit:
        evals_end = time.time()
        print("Newton eigenvalues took {:g} sec".format(evals_end-evals_start))

    return lam, np.reshape(np.asarray(v),n)
