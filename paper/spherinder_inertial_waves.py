import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import itertools
import multiprocessing as mp

import os
import pickle

import greenspan_inertial_waves as greenspan

from spherinder import config
config.default_normalize = False

import spherinder.operators as sph
from spherinder.eigtools import eigsort, plot_spectrum
from fileio import save_data, save_figure, plotspy


g_file_prefix = 'spherinder_inertial_waves'


def matrices(m, Lmax, Nmax, boundary_method):
    """Construct matrices for X = [i*u(+), i*u(-), i*u(z), p]"""
    alpha_bc = 0

    ncoeff = sum(sph.Nsizes(Lmax, Nmax))
    Zero = sparse.lil_matrix((ncoeff,ncoeff))
    I = sparse.eye(ncoeff)

    # Scalar gradient operators
    Gradp, Gradm, Gradz = sph.operator('grad')(m, Lmax, Nmax, alpha=0)

    # Vector divergence operator
    Div = sph.operator('div')(m, Lmax, Nmax, alpha=1)
    Divp, Divm, Divz = Div[:,:ncoeff], Div[:,ncoeff:2*ncoeff], Div[:,2*ncoeff:]

    # Boundary condition
    Rad = sph.operator('rdot', dtype='float128')(m, Lmax, Nmax, alpha=1)
    Boundary = sph.operator('boundary', dtype='float128', internal='float128')(m, Lmax, Nmax, alpha=1, sigma=0)
    Bound = Boundary @ Rad
    Bound = sph.remove_zero_rows(Bound).astype('float64')

    # Tau conversion
    # Time derivative matrices
    M00 = I
    M11 = I
    M22 = I
    M33 = Zero

    # i*u+ equation - spin+ velocity component
    L00 = 2 * I
    L01 = Zero
    L02 = Zero
    L03 = Gradp

    # i*u- equation - spin- velocity component
    L10 = Zero
    L11 = -2 * I
    L12 = Zero
    L13 = Gradm

    # i*w equation - vertical velocity component
    L20 = Zero
    L21 = Zero
    L22 = Zero
    L23 = Gradz

    # Divergence equation
    L30 = Divp
    L31 = Divm
    L32 = Divz
    L33 = Zero

    Mmats = [M00, M11, M22, M33]
    upmats = [L00, L01, L02, L03]
    ummats = [L10, L11, L12, L13]
    uzmats = [L20, L21, L22, L23]
    pmats = [L30, L31, L32, L33]

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([upmats, ummats, uzmats, pmats], format=sparse_format)

    # Boundary conditions
    def impenetrable():
        Nlengths = sph.Nsizes(Lmax, Nmax)
        Noffsets = np.append(0, np.cumsum(Nlengths)[:-1])

        ntau, npressure = np.shape(Bound)[0], np.sum(Nlengths)
        row = sparse.hstack([Bound, sparse.lil_matrix((ntau,npressure))])

        Conv = sph.convert_alpha(1-alpha_bc, m, Lmax, Nmax, alpha=alpha_bc, sigma=+1)
        col1 = sparse.hstack([Conv[:,Noffsets[ell]+Nlengths[ell]-1] for ell in range(Lmax-2)])
        col2 = Conv[:,Noffsets[-2]:]
        col = sparse.hstack([col1,col2])
        col = sparse.vstack([(i==0)*col for i in range(4)])

        return row, col

    # Create the boundary condition rows and tau columns
    row, col = impenetrable()

    corner = np.zeros((np.shape(row)[0], np.shape(col)[1]))
    L = sparse.bmat([[  L, col],
                     [row, corner]], format='csr')

    M = sparse.bmat([[    M, 0*col],
                     [0*row, corner]], format='csr')

    M, L = M.tocsr(), L.tocsr()
    return M, L


def filename_prefix(directory='data'):
    basepath = os.path.join(os.path.dirname(__file__), directory)
    prefix = g_file_prefix
    return os.path.join(basepath, os.path.join(prefix, prefix))


def pickle_filename(m, Lmax, Nmax, boundary_method, directory='data'):
    truncstr = '-truncated'
    return filename_prefix(directory) + f'-evalues-m={m}-Lmax={Lmax}-Nmax={Nmax}-{boundary_method}{truncstr}.pckl'


def solve_eigenproblem(m, Lmax, Nmax, boundary_method, plot_spy):
    # Construct the system
    M, L = matrices(m, Lmax, Nmax, boundary_method)

    if plot_spy:
        fig, plot_axes = plotspy(L, M)
        truncstr = '-truncated'
        filename = filename_prefix('figures') + f'-m={m}-Lmax={Lmax}-Nmax={Nmax}{truncstr}-spy.png'
        save_figure(filename, fig)
        plt.show()

    # Compute the eigenvalues and eigenvectors
    print('Eigenvalue problem, size {}'.format(np.shape(L)))
    evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)
    evalues = -evalues  # Flip the sign to match Greenspan

    # Output data
    data = {'m': m, 'Lmax': Lmax, 'Nmax': Nmax, 
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors}
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    save_data(filename, data)


def expand_evectors(Lmax, Nmax, vec, s, eta, bases):
    bases = [bases['up'], bases['um'], bases['uz'], bases['p']]

    # Extract the coefficients
    nfields = 4
    ncoeffs = [b.ncoeffs for b in bases]
    offsets = np.append(0, np.cumsum(ncoeffs))
    vec = vec.astype(np.complex128)
    vec[:offsets[3]] /= 1j
    coeffs = [vec[offsets[i]:offsets[i+1]] for i in range(nfields)]
    tau = vec[offsets[nfields]:]

    # Convert to grid space
    up, um, uz, p = tuple(bases[i].expand(coeffs[i]) for i in range(nfields))
    u, v, w = np.sqrt(0.5)*(up + um), -1j*np.sqrt(0.5)*(up - um), uz

    ss, ee = s[np.newaxis,:], eta[:,np.newaxis]
    ur = ss * u + ee * np.sqrt(1-ss**2) * w 

    return u, v, w, p, tau, ur


def plot_spectrum_callback(index, evalues, evectors, m, Lmax, Nmax, s, eta, bases):
    evalue, evector = evalues[index], evectors[:,index]
    u, v, w, p, tau, ur = expand_evectors(Lmax, Nmax, evector, s, eta, bases)

    field_indices = [3,4]
    fields = [u,v,w,p,ur]
    field_names = ['u','v','w','p','u_r']

    print('Tau norm: {}'.format(np.linalg.norm(tau)))
    if np.linalg.norm(tau) > .99:
        print('Tau norm too large - bad solution')

    check_boundary = True
    plot_boundary = False
    if check_boundary:
        z = eta[:,np.newaxis] * np.sqrt(1 - s[np.newaxis,:]**2)
        bc = ur
        bctop = bc[-1,:]
        bcbot = bc[ 0,:]
        error = [np.max(abs(b)) for b in [bctop, bcbot]]
        print(f'Boundary error: {error}')

        if plot_boundary:
            fig, ax = plt.subplots()
            ax.semilogy(s, abs(bctop), label='top')
            ax.semilogy(s, abs(bcbot), label='bottom')
            ax.set_xlabel('s')
            ax.set_title('Radial velocity on the boundary')
            ax.legend()
            ax.grid(True)
            fig.show()

    fig, ax = plt.subplots(1,len(field_indices),figsize=(7,4.5))
    for i in range(len(field_indices)):
        field_index = field_indices[i]
        f = fields[field_index]
        f = f.real if np.linalg.norm(f.real) >= np.linalg.norm(f.imag) else f.imag
        sph.plotfield(s, eta, f, fig=fig, ax=ax[i], colorbar=True)
        ax[i].set_title(r'${}$'.format(field_names[field_index]))
        if i > 0:
            ax[i].set_yticklabels([])
            ax[i].set_ylabel(None)

    fig.suptitle('λ = {:1.4f}'.format(evalue))
    fig.set_tight_layout(True)
    fig.show()


def plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_fields):
    save_plots = True
    if m > 30:
        n = m+61
    else:
        n = 60
    num_modes = 6
    modes = list(zip([n]*num_modes, range(num_modes)))

    # Load the data
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    if save_plots:
        def save(fn, fig): save_figure(fn, fig)
    else:
        def save(fn, fig): pass

    truncstr = '-truncated'
    configstr = f'm={m}-Lmax={Lmax}-Nmax={Nmax}-{boundary_method}{truncstr}'
    prefix = filename_prefix('figures')

    nlarge = len(np.where(np.abs(evalues.real) > 2)[0])
    indices = np.where(np.abs(evalues.real) <= 2)[0]
    evalues_in_range = evalues[indices]
    nimag = len(np.where(np.abs(evalues_in_range.imag) > 1e-15)[0])
    maximag = np.max(np.abs(evalues_in_range.imag))
    print('Number of bad eigenvalues: {}/{}'.format(nimag+nlarge,len(evalues)))
    print('    - Number out of real interval [-2,2]: {}/{}'.format(nlarge,len(evalues)))
    if nlarge > 0:
        maxind = np.argmax(np.abs(evalues.real))
        print('        - Maximum real part: {}'.format(evalues[maxind].real))
    print('    - Number with nonzero imaginary part: {}/{}'.format(nimag,len(evalues)))
    if nimag > 0:
        print('        - Maximum imaginary part: {}'.format(maximag))

    # Construct the bases
    ns, neta = 256, 257
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    bases = {'up': sph.Basis(s, eta, m, Lmax, Nmax, sigma=+1, alpha=1),
             'um': sph.Basis(s, eta, m, Lmax, Nmax, sigma=-1, alpha=1),
             'uz': sph.Basis(s, eta, m, Lmax, Nmax, sigma=0, alpha=1),
             'p':  sph.Basis(s, eta, m, Lmax, Nmax, sigma=0, alpha=0)}

    evalues, evectors = evalues[indices], evectors[:,indices]
    onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, m, Lmax, Nmax, s, eta, bases)

    # Plot the eigenvalues
    if plot_evalues:
        fig, ax = plot_spectrum(evalues, onpick=onpick)
        ax.set_xlim([-2.1,2.1])
        ax.set_title(f'Inertial Wave Eigenvalues, $m$ = {m}')

        filename = prefix + '-evalues-' + configstr + '.png'
        save(filename, fig)

    if not plot_fields:
        return

    # Get the target eigenpair
    fig, ax = plt.subplots(1,len(modes),figsize=(3*num_modes-1,5))
    for i, (n, ell) in enumerate(modes):
        # Compute the analytic eigenfrequency
        mode_index = (n,(n-m)//2-ell,m)
        evalue_target = 2*greenspan.compute_eigenvalues(mode_index[0], mode_index[2])[mode_index[1]-1]
        index = np.argmin(abs(evalues - evalue_target))
        evalue, evector = evalues[index], evectors[:,index]
    
        print('Plotting eigenvector with eigenvalue {:1.4f}'.format(evalue))
        u, v, w, p, tau, ur = expand_evectors(Lmax, Nmax, evector, s, eta, bases)
   
        # Plot the pressure field
        Fgrid = p
        relative_real = np.linalg.norm(np.real(Fgrid))/np.linalg.norm(Fgrid)
        f = Fgrid.real if relative_real > 0.5 else Fgrid.imag
        sph.plotfield(s, eta, f, colorbar=False, fig=fig, ax=ax[i])
        ax[i].set_title(f'λ/Ω = {evalue_target:.4f}i')
        if i > 0:
            ax[i].set_yticklabels([])
            ax[i].set_ylabel(None)

        # Compute the analytic pressure mode
        ss, ee = s[np.newaxis,:], eta[:,np.newaxis]
        zz = ee * np.sqrt(1 - ss**2)
        k = mode_index[1]
        panalytic = greenspan.compute_eigenmode(ss, zz, n, k, m)

        p /= np.max(abs(p))
        if np.max(abs(panalytic-p)) > np.max(abs(panalytic+p)):
            p *= -1
    
        # Compute the analytic eigenfrequency
        error_evalue = evalue_target - evalue
        print('    Eigenvalue error: ', error_evalue)

        error = panalytic - p.real
        print('    Eigenfunction error: ',np.max(abs(error)))

        # Spherical radial velocity component
        bc_error_top = np.max(np.abs(ur[-1,:]))
        bc_error_bot = np.max(np.abs(ur[ 0,:]))
        print('    Top    boundary error: {:1.3e}'.format(bc_error_top))
        print('    Bottom boundary error: {:1.3e}'.format(bc_error_bot))

        # Error plot
        plot_error = False
        if plot_error:
            sph.plotfield(s.ravel(), eta.ravel(), panalytic, colorbar=False)
            plt.title('analytic mode')
        
            sph.plotfield(s.ravel(), eta.ravel(), error)
            plt.title('error')

    filename = prefix + '-evectors-' + configstr + f'-n={n}-p.png'
    save(filename, fig)


def analyze_resolution():
    m = 30
    boundary_method = 'tau'

    if m == 95:
        n = m+61
        Lmax_values = [14,18,22,26,30]
        Nmax_values = [28,32,36,40,44,48,52,56,60]
    else:
        n = 60
        Lmax_values = [10,14,18,22,26]
        Nmax_values = [12,16,20,24,28,32,36,40,44,48,52,56,60]
    mode_targets = [(n,(n-m)//2-i,m) for i in range(4)]
    evalue_targets = [2*greenspan.compute_eigenvalues(i[0], i[2])[i[1]-1] for i in mode_targets]

    errors = np.zeros((len(evalue_targets),len(Lmax_values),len(Nmax_values)))
    for i,Lmax in enumerate(Lmax_values):
        for j,Nmax in enumerate(Nmax_values):
            # Load the data
            filename = pickle_filename(m, Lmax, Nmax, boundary_method)
            data = pickle.load(open(filename, 'rb'))
            evalues, evectors = data['evalues'], data['evectors']

            for k,evalue_target in enumerate(evalue_targets):
                # Compute the eigenvalue error
                index = np.argmin(abs(evalues - evalue_target))
                evalue, evector = evalues[index], evectors[:,index]
                error = evalue-evalue_target
                errors[k,i,j] = abs(error)


    prefix = filename_prefix('figures')

    for k,evalue in enumerate(evalue_targets):
        mode_prefix = prefix + f'-m={m}-evalue={evalue:1.2f}'

        fig, ax = plt.subplots()

        # Plotting niceties
        markers = ['s','o','d','^','X','h','p','P','*','v','<','>']
        for i,Lmax in enumerate(Lmax_values):
            marker = markers[i]
            ax.semilogy(Nmax_values, errors[k,i,:], f'-{marker}', label=f'Lmax={Lmax}')

        ax.legend(loc='upper left')
        ax.grid(True)
        ax.set_title(f'Eigenvalue Error, m = {m}, λ = {evalue:1.4f}')
        ax.set_xlabel('Nmax')
        ax.set_ylabel('Error')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save the figure
        filename = mode_prefix + f'-evalue_error.png'
        save_figure(filename, fig)

    plt.show()


def _solve_helper(m, Lmax, Nmax):
    boundary_method = 'tau'

    # Skip if we already have it
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    if os.path.exists(filename):
        print('  Already solved')
        return

    solve_eigenproblem(m, Lmax, Nmax, boundary_method, plot_spy=False)


def solve():
    m_values = [30]
    Lmax_values = [10,14,18,22,26,30]
    Nmax_values = [12,16,20,24,28,32,36,40,44,48,52,56,60]
    configs = itertools.product(m_values,Lmax_values,Nmax_values)

    pool = mp.Pool(mp.cpu_count()-1)
    pool.starmap(_solve_helper, configs)


def main():
    solve = True
    plot_evalues = True
    plot_fields = True
    plot_spy = False

    m = 95
    Lmax, Nmax = 12, 64
    boundary_method = 'tau'

    print(f'Inertial Waves, m = {m}')
    print(f'  Domain size: Lmax = {Lmax}, Nmax = {Nmax}')

    if solve:
        solve_eigenproblem(m, Lmax, Nmax, boundary_method, plot_spy)

    if plot_fields or plot_evalues:
        plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_fields)
        plt.show()


if __name__=='__main__':
    main()
#    solve()
#    analyze_resolution()

