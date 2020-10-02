import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy import sparse
import dedalus.public as de
import dedalus.tools.sparse as dts


@de.matsolvers.add_solver
class LSQR_solve(de.matsolvers.SparseSolver):
    """LSQR solve."""

    def __init__(self, matrix, solver=None):
        self.matrix = matrix.copy()

    def solve(self, vector):
        n = np.prod(np.shape(vector))
        vector = np.reshape(np.asarray(vector.ravel()), n)
        solution = sparse.linalg.lsqr(self.matrix, vector, atol=1e-14, btol=1e-14)
        return np.reshape(solution[0], (np.shape(self.matrix)[1], 1))



def eigsort(A, B, profile=False, overwrite=False, cutoff=np.inf):
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


matsolvers = ['UmfpackSpsolve', 'SuperluNaturalSpsolve', 'SuperluColamdSpsolve',
              'UmfpackFactorized', 'SuperluNaturalFactorized', 'SuperluColamdFactorized',
              'ScipyBanded', 'SPQR_solve', 'BandedQR', 'SparseInverse',
              'DenseInverse', 'BlockInverse']


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
        index = 1
        matsolver = de.matsolvers.matsolvers[matsolvers[index].lower()]

    if profile:
        print("  Starting eigenvalue computation at {}...".format(datetime.datetime.now()), flush=True)
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
        print("  Eigenvalues took {:g} sec".format(evals_end-evals_start), flush=True)

    return vals, vecs


def track_eigenpair(A, B, lam, v, matsolver=None, tol=1e-14, maxiter=10, verbose=False, profile=False):
    """Find a generalized eigenpair given initial guess (lam, v)
    """
    shape = np.shape(v)
    n = np.prod(shape)
    v = np.reshape(np.asmatrix(v), (n,1))

    # Normalize the initial guess
    v /= np.linalg.norm(v)

    if isinstance(matsolver, str):
        matsolver = de.matsolvers.matsolvers[matsolver.lower()]

    if profile:
        evals_start = time.time()

    for i in range(maxiter):
        M00 = A - lam * B
        M01 = - B @ v
        M10 = v.H
        M11 = np.zeros((1,1))

        r0 = M00 @ v
        r1 = 0.5 * (v.H @ v - 1)

        M = sparse.bmat([[M00, M01], [M10, M11]], format='csr')
        r = sparse.bmat([[r0],[r1]]).todense()

        if matsolver is not None:
            solver = matsolver(M)
            delta = solver.solve(-r)
        else:
            # Sparse solve
            delta = sparse.linalg.spsolve(M, -r)

        v[:, 0] += np.asmatrix(delta).T[:n, 0]
        lam += np.asscalar(delta[n])

        resid = np.linalg.norm(A @ v - lam * B @ v)
        if verbose:
            print('    Newton Iteration: {:3d},  residual norm: {:1.6e}'.format(i+1, resid))
        if resid <= tol:
            break

    if profile:
        evals_end = time.time()
        print('  Newton eigenvalues took {:g} sec'.format(evals_end-evals_start))

    return lam, np.reshape(v, shape)


def discard_spurious_eigenvalues(evalues, evalues_hires, cutoff=1e6, plot=False):
    """
    Solves the linear eigenvalue problem for two different resolutions.
    Returns trustworthy eigenvalues using nearest delta, from Boyd chapter 7.
    """
    lambda1 = evalues
    lambda2 = evalues_hires

    # Make sure argsort treats complex infs correctly
    for i in range(len(lambda1)):
        if np.isnan(lambda1[i]) or np.isinf(lambda1[i]):
            lambda1[i] = None
    for i in range(len(lambda2)):
        if np.isnan(lambda2[i]) or np.isinf(lambda2[i]):
            lambda2[i] = None

    # Sort lambda1 and lambda2 by real parts
    lambda1_indx = np.argsort(lambda1.real)
    lambda1 = lambda1[lambda1_indx]
    lambda2_indx = np.argsort(lambda2.real)
    lambda2 = lambda2[lambda2_indx]

    # try using lower res (gridnum = N1) instead
    sigmas = np.zeros(len(lambda1))
    sigmas[0] = np.abs(lambda1[0] - lambda1[1])
    sigmas[1:-1] = [0.5 * (np.abs(lambda1[j] - lambda1[j - 1]) + np.abs(lambda1[j + 1] - lambda1[j])) for j in
                    range(1, len(lambda1) - 1)]
    sigmas[-1] = np.abs(lambda1[-2] - lambda1[-1])

    # Ordinal delta, calculated for the number of lambda1's.
    delta_ord = np.abs(lambda1 - lambda2[:len(lambda1)]) / sigmas

    # Nearest delta
    delta_near = [np.nanmin(np.abs(lambda1[j] - lambda2)) for j in range(len(lambda1))] / sigmas

    if plot:
        plt.figure()
        plt.semilogy(1/delta_near, '.', marker='o', label=r'$δ_{nearest}$')
        plt.semilogy(1/delta_ord,  '.', marker='x', label=r'$δ_{ordinal}$')
        plt.xlabel('Mode Number')
        plt.grid(True)
        plt.title('Reciprocal eigenvalue drift ratios')
        plt.legend()

    # Discard eigenvalues with 1/delta_near < cutoff
    goodevals = lambda1[1 / delta_near > cutoff]

    return goodevals

