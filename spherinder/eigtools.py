import time, datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy import sparse
import scipy.sparse.linalg as spla
import dedalus.tools.array as dts

from . import matsolvers


def eigsort(A, B, profile=False, overwrite=False, cutoff=np.inf):
    """
    Compute the generalized eigenvalues for the system A.x = λ B.x,
    returning them sorted by real part.

    Parameters
    ----------
    A, B : numpy matrices
        Dense matrices for generalized eigenvalue problem
    profile : bool, optional
        Flag to time the eigenvalue computation
    overwrite : bool, optional
        Flag to perform computations in place in the B argument.
        May be a performance benefit but destroys the input array
    cutoff : float, optional
        Eigenvalues with larger modulus than the cutoff are thrown out

    Returns
    -------
    eigenvalues : np.ndarray
        Array with sorted eigenvalues
    eigenvectors : np.matrix
        Matrix with columns the generalized eigenvectors, sorted as eigenvalues
    """
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
    profile : bool, optional
        Flag to time the eigenvalue computation

    Other keyword options passed to scipy.sparse.linalg.eigs.
    """
    matsolver = matsolvers.make_solver(matsolver)

    if profile:
        print("  Starting eigenvalue computation at {}...".format(datetime.datetime.now()), flush=True)
        evalues_start = time.time()

    # Run dedalus wrapper on scipy sparse eigensolver
    evalues, evectors = dts.scipy_sparse_eigs(A, B, N, target, matsolver)

    # Remove infinite eigenvalues
    cutoff = np.inf
    bad = np.abs(evalues) > cutoff
    evalues[bad] = np.nan
    evectors = evectors[:,np.isfinite(evalues)]
    evalues = evalues[np.isfinite(evalues)]

    # Sort by real part
    i = np.argsort(evalues.real)
    evalues, evectors = evalues[i], evectors[:, i]

    if profile:
        evalues_end = time.time()
        print("  Eigenvalues took {:g} sec".format(evalues_end-evalues_start), flush=True)

    return evalues, evectors


def track_eigenpair(A, B, lam, v, matsolver=None, tol=1e-14, maxiter=10, verbose=False, profile=False):
    """
    Find a generalized eigenpair for the system A.x = λ B.x given initial guess
    (lam, v).  The routine uses Newton updates on the initial guess to refine
    the eigenpair.

    Parameters
    ----------
    A, B : scipy sparse matrices
        Sparse matrices for generalized eigenvalue problem
    lam : complex
        Initial eigenvalue guess
    vec : np.ndarray
        Initial eigenvector guess
    matsolver : str or matrix solver class, optional
        Matrix solver used to solve systems of the form A.x = b
    tol : float, optional
        Tolerance in 2-norm for the residual vector to stop iteration
    maxiter : int, optional
        Maximum number of iterations
    verbose : bool, optional
        Flag to print convergence messages each iteration
    profile : bool, optional
        Flag to time the eigenvalue computation

    Returns
    -------
    eigenvalue : complex
        Refined eigenvalue
    eigenvector : np.array
        Refined eigenvector
    """
    shape = np.shape(v)
    n = np.prod(shape)
    v = np.reshape(np.asmatrix(v), (n,1))

    # Normalize the initial guess
    v /= np.linalg.norm(v)

    if isinstance(matsolver, str):
        matsolver = matsolvers.make_solver(matsolver)

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
            delta = spla.spsolve(M, -r)

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

    Parameters
    ----------
    evalues : np.ndarray
        Eigenvalues from the low resolution solution
    evalues_hires : np.ndarray
        Eigenvalues from the high resolution solution
    cutoff : float, optional
        Cutoff for testing whether an eigenvalue is resolved.
        Larger values are more restrictive
    plot : bool, optional
        Flag to plot the two sets of eigenvalues

    Returns
    -------
    good_eigenvalues : np.ndarray
        Array of well-resolved eigenvalues
    indices : np.ndarray
        Indices of the resolved eigenvalues in the low resolution evalues array
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
        plt.semilogy(1/delta_near, '.', marker='o', markersize=1.5, label=r'$δ_{nearest}$')
        plt.semilogy(1/delta_ord,  '.', marker='x', markersize=1.5, label=r'$δ_{ordinal}$')
        plt.xlabel('Mode Number')
        plt.grid(True)
        plt.title('Reciprocal eigenvalue drift ratios')
        plt.legend()

    # Discard eigenvalues with 1/delta_near < cutoff
    indices = np.asarray(1/delta_near > cutoff).nonzero()[0]
    goodevals = lambda1[indices]

    return goodevals, indices


def plot_spectrum(evalues, onpick=None, figax=None, *args, **kwargs):
    """
    Standard eigenvalue plotting routine with an optional callback when
    an eigenvalue is selected.  This enables plotting the associated
    eigenvector with a custom routine.

    Parameters
    ----------
    evalues : numpy.ndarray
        Array of (possibly complex) eigenvalues
    onpick : function(int)
        Function called when an eigenvalue is selected.  The corresponding
        index is passed to the routine for further analysis.

    Other arguments and keyword options passed to matplotlib.pyplot.Axes.plot

    Returns
    -------
    fig, ax : matplotlib.pyplot Figure and Axes objects
    """
    def onpick_callback(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        point = tuple(zip(xdata[ind], ydata[ind]))[0]
        evalue = point[0] + 1j*point[1]
        print('selected eigenvalue: {}'.format(evalue))

        if onpick is not None:
            index = np.argmin(np.abs(evalues-evalue))
            onpick(index)
        return True

    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    ax.plot(evalues.real, evalues.imag, '.', markersize=2, picker=True, pickradius=5, *args, **kwargs)
    ax.set_xlabel('Real(λ)')
    ax.set_ylabel('Imag(λ)')
    ax.grid()
    fig.canvas.mpl_connect('pick_event', onpick_callback)

    return fig, ax

