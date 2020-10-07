import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from eigtools import eigsort
import os
import pickle

from dedalus_sphere import jacobi as Jacobi

import config
config.internal_dtype = 'float64'
import spherinder as sph


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


def analytic_evalues(m, L, N):
    return [dispersion_zeros(m+ell, N) for ell in range(L)]


def matrices(m, Lmax, Nmax, boundary_method, truncate=True):
    """Construct matrices for Bessel's eigenproblem, Lap(f) + λ f = 0
    """
    alpha = 0
    alpha_bc = alpha+1
    alpha_bc_n = alpha+1
    alpha_bc_trunc = alpha+2
    trunc_kind = 'beta'  # one of [alpha, beta]

    # Differentiation matrices
    M = -sph.convert_alpha_up_n(2, m, Lmax, Nmax, alpha=alpha, sigma=0, truncate=truncate)
    L = sph.operator('laplacian')(m, Lmax, Nmax, alpha=alpha)

    # Resize matrices as needed
    if truncate:
        L = sph.resize(L, Lmax, Nmax+1, Lmax, Nmax)
    else:
        n_for_ell = lambda ell: Nmax+1 if ell < Lmax-2 else Nmax
        M = sph.resize(M, Lmax, Nmax+1, Lmax, n_for_ell)
        L = sph.resize(L, Lmax, Nmax+1, Lmax, n_for_ell)

    # Boundary condition in eta direction
    Conv = sph.convert_alpha_up_n(alpha+2-alpha_bc, m, Lmax, Nmax, alpha=alpha_bc, sigma=0, truncate=truncate)
    if not truncate:
        Conv = sph.resize(Conv, Lmax, Nmax+1, Lmax, n_for_ell)
    col1 = Conv[:,-2*Nmax:]

    # Boundary condition in s direction
    Conv = sph.convert_alpha_up_n(alpha+2-alpha_bc_n, m, Lmax, Nmax, alpha=alpha_bc_n, sigma=0, truncate=truncate)
    if not truncate:
        Conv = sph.resize(Conv, Lmax, Nmax+1, Lmax, n_for_ell)
    col2 = Conv[:,Nmax-1:-2*Nmax:Nmax]

    col = sparse.hstack([col1,col2])

    # Additional tau columns for non-truncation
    if not truncate:
        if trunc_kind == 'alpha':
            Conv = sph.convert_alpha_up_n(alpha+2-alpha_bc_trunc, m, Lmax, Nmax, alpha=alpha_bc_trunc, sigma=0, truncate=truncate)
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


def checkdir(filename):
    filename = os.path.abspath(filename)
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.mkdir(path)

def savedata(filename, data):
    checkdir(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def savefig(filename):
    checkdir(filename)
    plt.savefig(filename)
    

def filename_prefix(directory='data'):
    basepath = os.path.join(os.path.dirname(__file__), directory)
    prefix = 'geometric_bessel'
    return os.path.join(basepath, os.path.join(prefix, prefix))


def pickle_filename(m, Lmax, Nmax, boundary_method, directory='data'):
    return filename_prefix(directory) + '-evalues-m={}-Lmax={}-Nmax={}-{}.pckl'.format(m,Lmax,Nmax,boundary_method)


def solve_eigenproblem(m, Lmax, Nmax, boundary_method):
    # Construct the system
    M, L = matrices(m, Lmax, Nmax, boundary_method)

    plot_spy = False
    if plot_spy:
        fig, plot_axes = plt.subplots(1,2,figsize=(9,4))
        plot_axes[0].spy(L)
        plot_axes[1].spy(M)
        plt.show()

    # Compute the eigenvalues and eigenvectors
    print('Eigenvalue problem, size {}'.format(np.shape(L)))
    evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)

    # Output data
    data = {'m': m, 'Lmax': Lmax, 'Nmax': Nmax, 
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors}
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    savedata(filename, data)


def expand_evectors(m, Lmax, Nmax, boundary_method, vec, s, eta):
    dtype = 'float64'

    # Get the grid space vector fields
    ncoeff = Lmax*Nmax
    fcoeff = vec[:ncoeff].real.reshape((Lmax,Nmax)).astype(dtype)
    tau = vec[ncoeff:]

    # Convert to grid space
    basis = [sph.psi(Nmax, m, ell, s, eta, sigma=0, alpha=0, dtype=dtype) for ell in range(Lmax)]
    f = sph.expand(basis, fcoeff)

#    f /= np.max(abs(f))
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
        def save(fn): savefig(fn)
    else:
        def save(_): pass

    evalue_target = 49.14**2
#    evalue_target = 112.60858**2
    configstr = 'm={}-Lmax={}-Nmax={}-{}'.format(m,Lmax,Nmax,boundary_method)
    prefix = filename_prefix('figures')

    ntotal = len(evalues)
    nimag = len(np.where(np.abs(evalues.imag) > 1e-15)[0])
    print(f'Number of complex eigenvalues: {nimag}/{ntotal}')

    check_boundary(m, Lmax, Nmax, evalues, evectors)

    # Plot the eigenvalues
    if plot_evalues:
        fig, ax = plt.subplots()
        ax.plot(evalues.real, evalues.imag, '.', markersize=3, color='tab:blue')
        ax.grid()
        ax.set_xlabel('Real(λ)')
        ax.set_ylabel('Imag(λ)')
        ax.set_title('Bessel Eigenvalues in the Stretched Sphere')
        fig.set_tight_layout(True)

        filename = prefix + '-evalues-' + configstr + '.png'
        save(filename)

    if not plot_fields:
        return

    # Get the target eigenpair
    index = np.argmin(abs(evalues - evalue_target))
    val, vec = evalues[index], evectors[:,index]

    kappa = np.sqrt(val.real)
    print('Plotting eigenvector with eigenvalue {:1.4f}'.format(kappa))

    # Construct the basis polynomials
    ns, neta = 1024, 1
    s, eta = np.linspace(0,1,ns+1)[1:], np.array([0.])
    f, tau = expand_evectors(m, Lmax, Nmax, boundary_method, vec, s, eta)

    # Spherical radial velocity component
    error_boundary = np.max(np.abs(f[0,-1]))
    print('Boundary error: {:1.3e}'.format(error_boundary))

    fields = [f]
    field_names = ['f']
    plot_field_indices = [0]

    for i in range(len(plot_field_indices)):
        field_index = plot_field_indices[i]
        Fgrid = fields[field_index]

        fig, ax = plt.subplots()
        ax.plot(s,f.ravel())
        ax.grid()
        ax.set_xlabel('s')
        ax.set_ylabel('f')
        ax.set_title(r'${}$'.format(field_names[field_index]))
        filename = prefix + '-evector-' + configstr + '-' + '-' + field_names[field_index] + '.png'
        save(filename)


def main():
    solve = True
    plot_evalues = True
    plot_fields = True

    m = 10
    Lmax, Nmax = 20, 30
    boundary_method = 'tau'

    print('Bessel Eigenproblem, m = {}'.format(m))
    print('  Domain size: Lmax = {}, Nmax = {}'.format(Lmax, Nmax))

    if solve:
        solve_eigenproblem(m, Lmax, Nmax, boundary_method)

    if plot_fields or plot_evalues:
        plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_fields)
        plt.show()


if __name__=='__main__':
    main()
