import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.special import spherical_jn
import itertools
import multiprocessing as mp

import os
import pickle

from dedalus_sphere import jacobi as Jacobi

from spherinder import config
config.internal_dtype = 'float64'

import spherinder.operators as sph
from spherinder.eigtools import eigsort, plot_spectrum
from fileio import save_data, save_figure, plotspy

g_file_prefix = 'spherinder_bessel'


def dispersion_zeros(ell,n,a=0,guess=None,imax=20,nk=10,eps=0.1):
    j = spherical_jn
    def F(k,deriv=False): 
        return j(ell,k,derivative=deriv) - a*j(ell+2,k,derivative=deriv)
    
    if guess == None:    
        kmax = np.pi*(n+ell/2 + eps)
        k = np.linspace(0,kmax,int(kmax*nk))
        S = np.sign(F(k))
        i = np.where(np.abs(np.roll(S,-1)-S)==2)[0]
        k = 0.5*(k[i]+k[i+1])
    else: k = guess
    
    for i in range(imax):
        dk =  F(k)/F(k,deriv=True)
        k -= dk
    
    return k


def matrices_tau(m, Lmax, Nmax, truncate=True):
    """Construct matrices for Bessel's eigenproblem, Lap(f) + λ f = 0
    """
    alpha = 0
    alpha_bc = alpha+1
    alpha_bc_n = alpha+1
    alpha_bc_trunc = alpha+2
    trunc_kind = 'beta'  # one of [alpha, beta]

    # Differentiation matrices
    M = -sph.convert_alpha(2, m, Lmax, Nmax, alpha=alpha, sigma=0, truncate=truncate)
    L = sph.operator('laplacian')(m, Lmax, Nmax, alpha=alpha)

    # Resize matrices as needed
    if truncate:
        L = sph.resize(L, Lmax, Nmax+1, Lmax, Nmax)
    else:
        n_for_ell = lambda ell: Nmax+1 if ell < Lmax-2 else Nmax
        M = sph.resize(M, Lmax, Nmax+1, Lmax, n_for_ell)
        L = sph.resize(L, Lmax, Nmax+1, Lmax, n_for_ell)

    # Boundary condition in eta direction
    Conv = sph.convert_alpha(alpha+2-alpha_bc, m, Lmax, Nmax, alpha=alpha_bc, sigma=0, truncate=truncate)
    if not truncate:
        Conv = sph.resize(Conv, Lmax, Nmax+1, Lmax, n_for_ell)
    col1 = Conv[:,-2*Nmax:]

    # Boundary condition in s direction
    Conv = sph.convert_alpha(alpha+2-alpha_bc_n, m, Lmax, Nmax, alpha=alpha_bc_n, sigma=0, truncate=truncate)
    if not truncate:
        Conv = sph.resize(Conv, Lmax, Nmax+1, Lmax, n_for_ell)
    col2 = Conv[:,Nmax-1:-2*Nmax:Nmax]

    col = sparse.hstack([col1,col2])

    # Additional tau columns for non-truncation
    if not truncate:
        if trunc_kind == 'alpha':
            Conv = sph.convert_alpha(alpha+2-alpha_bc_trunc, m, Lmax, Nmax, alpha=alpha_bc_trunc, sigma=0, truncate=truncate)
            Conv = sph.resize(Conv, Lmax, Nmax+1, Lmax, n_for_ell)
            col3 = Conv[:,Nmax-2:-2*Nmax:Nmax]
        elif trunc_kind == 'beta':
            zmat = sparse.eye(Lmax)
            A = Jacobi.operator('A')
            smats = [(A(+1)**(alpha+2-alpha_bc_trunc))(Nmax+1,ell+alpha_bc_trunc+1/2,m) for ell in range(Lmax)]
            op = sph.make_operator(zmat, smats)
            op = sph.resize(op, Lmax, Nmax+1, Lmax, n_for_ell)
            col3 = op[:,Nmax:-2*(Nmax+1):Nmax+1]
        else:
            raise ValueError('Unrecognized boundary kind')
        col = sparse.hstack([col,col3])

    # Boundary condition
    row = sph.operator('boundary')(m, Lmax, Nmax, alpha=alpha, sigma=0)

    # Create the boundary condition rows and tau columns
    corner = np.zeros((np.shape(row)[0], np.shape(col)[1]))
    L = sparse.bmat([[  L, col],
                     [row, corner]], format='csr')

    M = sparse.bmat([[    M, 0*col],
                     [0*row, corner]], format='csr')

    M, L = M.tocsr(), L.tocsr()
    return M, L


def matrices_galerkin(m, Lmax, Nmax, truncate=True):
    alpha_bc = 1
    Lout, Nout = Lmax+2, Nmax+1

    M = -sph.convert_alpha(2, m, Lout, Nout, alpha=0, sigma=0, truncate=truncate)
    L = sph.operator('laplacian')(m, Lout, Nout, alpha=0)
    if truncate:
        L = sph.resize(L, Lout, Nout+1, Lout, Nout)

    # Multiplication by 1-r**2 lowers alpha by 1
    Bound = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=1, sigma=0)

    M = M @ Bound
    L = L @ Bound

    Conv = sph.convert_alpha(2-alpha_bc, m, Lout, Nout, alpha=alpha_bc, sigma=0, truncate=truncate)
    col1, col2 = Conv[:,Nout-1:-2*Nout:Nout], Conv[:,-2*Nout:]

    L = sparse.hstack([L,  col1,  col2])
    M = sparse.hstack([M,0*col1,0*col2])

    return M, L


def matrices(m, Lmax, Nmax, boundary_method):
    if boundary_method in ['tau', 'galerkin']:
        return eval('matrices_' + boundary_method)(m, Lmax, Nmax)
    else:
        raise ValueError('Unsupported boundary method')


def filename_prefix(directory='data'):
    basepath = os.path.join(os.path.dirname(__file__), directory)
    prefix = g_file_prefix
    return os.path.join(basepath, os.path.join(prefix, prefix))


def pickle_filename(m, Lmax, Nmax, boundary_method, directory='data', ext='.pckl', prefix='evalues'):
    return filename_prefix(directory) + f'-{prefix}-m={m}-Lmax={Lmax}-Nmax={Nmax}-{boundary_method}{ext}'


def solve_eigenproblem(m, Lmax, Nmax, boundary_method):
    # Construct the system
    M, L = matrices(m, Lmax, Nmax, boundary_method)

    plot_spy = False
    if plot_spy:
        fig, plot_axes = plotspy(L, M)
        filename = pickle_filename(m, Lmax, Nmax, boundary_method, directory='figures', ext='.png', prefix='spy')
        save_figure(filename, fig)

    # Compute the eigenvalues and eigenvectors
    print('Eigenvalue problem, size {}'.format(np.shape(L)))
    evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)

    # Output data
    data = {'m': m, 'Lmax': Lmax, 'Nmax': Nmax, 
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors}
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    save_data(filename, data)


def expand_evectors(m, Lmax, Nmax, boundary_method, vec, s, eta):
    dtype = 'float64'

    # Get the grid space vector fields
    ncoeff = Lmax*Nmax
    fcoeff = vec[:ncoeff].real.reshape((Lmax,Nmax)).astype(dtype)
    tau = vec[ncoeff:]
#    print('Tau norm: {}'.format(np.linalg.norm(tau)))

    # Convert to grid space
    if boundary_method == 'galerkin':
        Conv = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=1, sigma=0)
        Lmax = Lmax+2
        Nmax = Nmax+1
        fcoeff = (Conv @ fcoeff.ravel()).reshape(Lmax, Nmax)

    basis  = sph.Basis(s, eta, m, Lmax, Nmax, sigma=0,  alpha=0, galerkin=False, dtype=dtype)
    f = basis.expand(fcoeff)

    return f, tau


def check_boundary(m, Lmax, Nmax, evalues, evectors, plot=False):
    bc = sph.operator('boundary')(m, Lmax, Nmax, alpha=0, sigma=0)
    result = bc @ evectors[:Nmax*Lmax,:]
    error = np.linalg.norm(result, axis=0)
    index = np.argmax(error)

    if plot:
        plt.figure()
        plt.semilogy(error)
        plt.xlabel('Mode Index')
        plt.ylabel('L2 coefficient error')
        plt.title('Boundary Error')
        plt.grid()
    print('Worst case coefficient error, index {}, evalue {}, L2 error {:1.4e}'.format(index, evalues[index], error[index]))


def plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_fields):
    save_plots = False

    # Load the data
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    bad = (np.abs(evalues) > 1e10)
    evalues[bad] = np.nan
    evectors = evectors[:,np.isfinite(evalues)]
    evalues = evalues[np.isfinite(evalues)]

    if save_plots:
        def save(fn, fig): save_figure(fn, fig)
    else:
        def save(fn, fig): pass

    mode_number = 10
    kappa_analytic = dispersion_zeros(m, mode_number+1)[mode_number]
    evalue_target = kappa_analytic**2

    configstr = 'm={}-Lmax={}-Nmax={}-{}'.format(m,Lmax,Nmax,boundary_method)
    prefix = filename_prefix('figures')

    ntotal = len(evalues)
    nimag = len(np.where(np.abs(evalues.imag) > 1e-15)[0])
    print(f'Number of complex eigenvalues: {nimag}/{ntotal}')

    if boundary_method == 'tau':
        check_boundary(m, Lmax, Nmax, evalues, evectors)

    # Plot the eigenvalues
    if plot_evalues:
        fig, ax = plot_spectrum(evalues)
        ax.set_title('Bessel Eigenvalues in the Stretched Sphere')

        filename = prefix + '-evalues-' + configstr + '.png'
        save(filename, fig)

    if not plot_fields:
        return

    # Get the target eigenpair
    index = np.argmin(abs(evalues - evalue_target))
    val, vec = evalues[index], evectors[:,index]

    kappa = np.sqrt(val.real)
    print('Plotting eigenvector with eigenvalue {:1.8f}'.format(kappa))

    evalue_error = (val.real-evalue_target)/evalue_target
    print('Relative Eigenvalue error: {:e}'.format(evalue_error))

    # Construct the basis polynomials
    ns, neta = 1024, 1
    s, eta = np.linspace(0,1,ns+1)[1:], np.array([0.])
    f, tau = expand_evectors(m, Lmax, Nmax, boundary_method, vec, s, eta)

    # Spherical radial velocity component
    error_boundary = np.max(np.abs(f[0,-1]))
    print('Boundary error: {:1.3e}'.format(error_boundary))

    f /= np.max(np.abs(f))
    if np.abs(np.min(f)) > np.max(f):
        f = -f
    fig, ax = plt.subplots()
    ax.plot(s, f.ravel())
    ax.grid()
    ax.set_xlabel('s')
    ax.set_ylabel('f')
    ax.set_title(r'$f$')
    filename = prefix + '-evector-' + configstr + '.png'
    save(filename, fig)

    fanalytic = spherical_jn(m, kappa_analytic*s)
    fanalytic /= np.max(np.abs(fanalytic))
    ferror = f - fanalytic

    print('Max eigenvector error: {}'.format(np.max(abs(ferror))))

    fig, ax = plt.subplots()
    ax.plot(s, ferror.ravel())
    ax.grid(True)
    ax.set_xlabel('s')
    ax.set_ylabel('error')

    # Plot the 2D field
    ns, neta = 200, 201
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    f, tau = expand_evectors(m, Lmax, Nmax, boundary_method, vec, s, eta)

    fig, ax = plt.subplots()
    sph.plotfield(s, eta, f, fig, ax)


def analyze_resolution():
    m = 30
    boundary_method = 'tau'

    if m == 2:
        modes = [4]
        Lmax_values = [20,25,30,40,50]
        Nmax_values = [5,10,15,20,25,30,40,50]
    else:
        modes = [10] #,16,20,24]
        Lmax_values = [30,40,50,60,70]
        Nmax_values = [20,25,30,40,50,60]
    kappa_targets = dispersion_zeros(m, max(modes)+1)[modes]

    # Compute the analytic modes
    ns, neta = 1024, 1
    s, eta = np.linspace(0,1,ns+1)[1:], np.array([0.])

    analytic_modes = np.zeros((ns,len(kappa_targets)))
    for k,kappa in enumerate(kappa_targets):
        fanalytic = spherical_jn(m, kappa*s)
        fanalytic /= np.max(np.abs(fanalytic))
        analytic_modes[:,k] = fanalytic


    errors_evalue = np.zeros((len(kappa_targets),len(Lmax_values),len(Nmax_values)))
    errors_mode = np.zeros((len(kappa_targets),len(Lmax_values),len(Nmax_values)))
    for i,Lmax in enumerate(Lmax_values):
        for j,Nmax in enumerate(Nmax_values):
            # Load the data
            filename = pickle_filename(m, Lmax, Nmax, boundary_method)
            data = pickle.load(open(filename, 'rb'))
            evalues, evectors = data['evalues'], data['evectors']

            for k,kappa in enumerate(kappa_targets):
                evalue_target = kappa**2

                # Compute the eigenvalue error
                index = np.argmin(abs(evalues - evalue_target))
                evalue, evector = evalues[index], evectors[:,index]
                error = abs(evalue-evalue_target)/evalue_target
                errors_evalue[k,i,j] = error

                # Compute the mode error
                f, _ = expand_evectors(m, Lmax, Nmax, boundary_method, evector, s, eta)
                f = f.reshape(ns)
                f /= np.max(np.abs(f))
                if abs(np.min(f)) > np.max(f):
                    f = -f

                error = np.max(np.abs(f-analytic_modes[:,k]))
                errors_mode[k,i,j] = error


    prefix = filename_prefix('figures')

    for k,kappa in enumerate(kappa_targets):
        mode_prefix = prefix + f'-m={m}-kappa={kappa:1.2f}'

        # Plot the analytic mode
        fig, ax = plt.subplots()
        ax.plot(s, analytic_modes[:,k])
        ax.set_title(f'Bessel Eigenmode: m = {m}, κ = {kappa:1.2f}')
        ax.set_xlabel('s')
        ax.grid(True)

        # Save the analytic mode
        filename = mode_prefix + '-analytic.png'
        save_figure(filename, fig)

        def plotfn(errors, which):
            # Plot the error in eigenvalue
            fig, ax = plt.subplots()

            # Plotting niceties
            markers = ['s','o','d','^','X','h','p','P','*','v','<','>']
            for i,Lmax in enumerate(Lmax_values):
                marker = markers[i]
                ax.semilogy(Nmax_values, errors[k,i,:], f'-{marker}', label=f'Lmax={Lmax}')

            loc = 'lower right' if boundary_method == 'tau' else 'upper right'
            ax.legend(loc=loc)
            ax.grid(True)
            title = 'Eigenvalue' if which == 'evalue' else 'Mode'
            bound = 'Galerkin' if boundary_method == 'galerkin' else 'Tau'
            ax.set_title(f'{title} Error, m = {m}, κ = {kappa:1.2f}, {bound}')
            ax.set_xlabel('Nmax')
            ax.set_ylabel('Error')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Save the figure
            filename = mode_prefix + f'-error-{which}-{boundary_method}.png'
            save_figure(filename, fig)

        plotfn(errors_evalue, 'evalue')
        plotfn(errors_mode, 'mode')

    plt.show()


def _solve_helper(m, Lmax, Nmax, boundary_method):
    print('Bessel Eigenproblem, m = {}, boundary method = {}'.format(m, boundary_method))
    print('  Domain size: Lmax = {}, Nmax = {}'.format(Lmax, Nmax))

    # Skip if we already have it
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    if os.path.exists(filename):
        print('  Already solved')
        return

    solve_eigenproblem(m, Lmax, Nmax, boundary_method)


def solve():
    solve = True

    m_values = [30]
    Lmax_values = [20,30,40,50,60,70,80]
    Nmax_values = [20,25,30,40,50,60]
    boundary_methods = ['tau','galerkin']
    configs = itertools.product(m_values,Lmax_values,Nmax_values,boundary_methods)

    pool = mp.Pool(mp.cpu_count())
    pool.starmap(_solve_helper, configs)


def main():
    solve = True
    plot_evalues = False
    plot_fields = True

    m = 10
    Lmax, Nmax = 50, 50
    boundary_method = 'galerkin'

    print('Bessel Eigenproblem, m = {}, boundary method = {}'.format(m, boundary_method))
    print('  Domain size: Lmax = {}, Nmax = {}'.format(Lmax, Nmax))

    if solve:
        solve_eigenproblem(m, Lmax, Nmax, boundary_method)

    if plot_fields or plot_evalues:
        plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_fields)
        plt.show()


if __name__=='__main__':
#    main()
#    solve()
    analyze_resolution()

