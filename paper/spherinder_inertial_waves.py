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
from spherinder.eigtools import eigsort, scipy_sparse_eigs, plot_spectrum
from fileio import save_data, save_figure, plotspy

import spherinder.asymptotic_operators as sao


g_file_prefix = 'spherinder_inertial_waves'
use_full_vertical_velocity = False


def vertical_velocity_size(Lmax, Nmax, full=use_full_vertical_velocity):
    return (Lmax, Nmax) if full else (Lmax-1, Nmax)


def matrices(m, Lmax, Nmax, boundary_method):
    """Construct matrices for X = [i*u(+), i*u(-), i*u(z), p]"""
    alpha_bc = 0

    def zeros(n,m):
        return sparse.lil_matrix((n,m))

    ncoeff = sph.num_coeffs(Lmax, Nmax)
    ncoeffw = sph.num_coeffs(*vertical_velocity_size(Lmax, Nmax))
    Zero = zeros(ncoeff,ncoeff)
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

    Nlengths, Noffsets = sph.coeff_sizes(Lmax, Nmax)
    if not use_full_vertical_velocity:
        Divz = Divz[:,:Noffsets[-1]]
        Gradz = Gradz[:Noffsets[-1],:]
        Bound = Bound[:,:2*ncoeff+Noffsets[-1]]

    # Tau conversion
    # Time derivative matrices
    M00 = I
    M11 = I
    M22 = sparse.eye(ncoeffw)
    M33 = Zero

    # i*u+ equation - spin+ velocity component
    L00 = 2 * I
    L01 = Zero
    L02 = zeros(ncoeff, ncoeffw)
    L03 = Gradp

    # i*u- equation - spin- velocity component
    L10 = Zero
    L11 = -2 * I
    L12 = zeros(ncoeff, ncoeffw)
    L13 = Gradm

    # i*w equation - vertical velocity component
    L20 = zeros(ncoeffw, ncoeff)
    L21 = zeros(ncoeffw, ncoeff)
    L22 = zeros(ncoeffw, ncoeffw)
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
        ntau, npressure = np.shape(Bound)[0], ncoeff
        row = sparse.hstack([Bound, sparse.lil_matrix((ntau,npressure))])
        col = sph.tau_projection(m, Lmax, Nmax, alpha=1, sigma=+1, alpha_bc=alpha_bc)
        col = sparse.vstack([col, sparse.lil_matrix((np.shape(L)[0]-np.shape(col)[0],np.shape(col)[1]))])

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


def pickle_filename(m, Lmax, Nmax, boundary_method, directory='data', evalue_target=None):
    evstr = '' if evalue_target is None else f'-λ={evalue_target:1.5f}'
    return filename_prefix(directory) + f'-evalues-m={m}-Lmax={Lmax}-Nmax={Nmax}-{boundary_method}{evstr}.pckl'


def solve_eigenproblem(m, Lmax, Nmax, boundary_method, plot_spy, evalue_target=None):
    # Construct the system
    M, L = matrices(m, Lmax, Nmax, boundary_method)

    if plot_spy:
        fig, plot_axes = plotspy(L, M)
        filename = filename_prefix('figures') + f'-m={m}-Lmax={Lmax}-Nmax={Nmax}-spy.png'
        save_figure(filename, fig)
        plt.show()

    # Compute the eigenvalues and eigenvectors
    print('Eigenvalue problem, size {}'.format(np.shape(L)))
    if evalue_target is None:
        evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)
        evalues = -evalues  # Flip the sign to match Greenspan
    else:
        matsolver = 'UmfpackFactorized'
        evalues, evectors = scipy_sparse_eigs(L, M, N=4, target=-evalue_target, matsolver=matsolver, profile=True)
        evalues = -evalues  # Flip the sign to match Greenspan

    # Output data
    data = {'m': m, 'Lmax': Lmax, 'Nmax': Nmax, 
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors}
    filename = pickle_filename(m, Lmax, Nmax, boundary_method, evalue_target=evalue_target)
    save_data(filename, data)


def expand_evectors(Lmax, Nmax, vec, s, eta, bases, return_coeffs=False):
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

    if return_coeffs:
        return u, v, w, p, tau, ur, coeffs
    else:
        return u, v, w, p, tau, ur


def plot_spectrum_callback(index, evalues, evectors, m, Lmax, Nmax, s, eta, bases):
    evalue, evector = evalues[index], evectors[:,index]
    u, v, w, p, tau, ur, coeffs = expand_evectors(Lmax, Nmax, evector, s, eta, bases, return_coeffs=True)

    fields = [u,v,w,p,ur]
    """
    field_indices = [3,4]
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
    """

    # Get grid and coefficient data
    field_index = 3
    fgrid, fcoeff = fields[field_index], coeffs[field_index]

    fnorm = np.linalg.norm(fcoeff)

    lengths, offsets = sph.coeff_sizes(Lmax, Nmax)
    pruned = []
    for ell in range(Lmax):
        final = offsets[ell]+lengths[ell]

        nprune = 2
        for n in range(nprune):
            pruned.append(fcoeff[final-nprune+n])
        fcoeff[final-nprune:final] = 0

    print(np.shape(pruned))
    print(f'Pruned norm: {np.linalg.norm(pruned)}, Vector Norm: {fnorm}')

    conv = sph.convert_alpha(2, m, Lmax, Nmax, alpha=0, sigma=0)
    fgrid = bases['p2'].expand(conv @ fcoeff)

    # Compute the Laplacian
    lap = sph.operator('laplacian')(m, Lmax, Nmax, alpha=0)
    flap = bases['p2'].expand(lap @ fcoeff)

    # Compute the horizontal Laplacian
    laph = sao.horizontal_laplacian(m, Lmax, Nmax, alpha=0)
    flaph = bases['p2'].expand(laph @ fcoeff)

    # Compute the d/dphi**2 part of the Laplacian
    fphiphi = -1/s**2 * m**2 * fgrid

    solns, names = zip(*[(fgrid, 'p'), (flap, 'Lap(p)'), (flaph, 'HLap(p)'), (fphiphi, 'PhiLap(p)')])

#    sindex = np.argmin(s < 0.7)
    sindex = len(s)

    # Plot them
    fig, ax = plt.subplots(1,4, figsize=(15,5))
    for i in range(4):
        f = abs(solns[i])
        sph.plotfield(s[:sindex], eta, f[:,:sindex], fig=fig, ax=ax[i], colorbar=True, stretch=True)
        ax[i].set_title(r'${}$'.format(names[i]))
        if i > 0:
            ax[i].set_yticklabels([])
            ax[i].set_ylabel(None)

    fig.set_tight_layout(True)
    fig.show()


def normalize_coeffs(c):
    cmin, cmax = np.min(c), np.max(c)
    scale = cmin if abs(cmin) > cmax else cmax
    return c / scale


def compute_coeff_error(bases, mode_index, evector):
    m, Lmax, Nmax = bases['p'].m, bases['p'].Lmax, bases['p'].Nmax

    def mode_fun(t, eta):
        s = np.sqrt((1+t)/2)
        z = np.sqrt(1-s**2)*eta[:,np.newaxis]
        return greenspan.compute_eigenmode(s, z, *mode_index, normalize=True)

    projected_coeffs = greenspan.project(mode_fun, m, Lmax, Nmax, alpha=0, sigma=0, Lquad=400, Nquad=400)
    offsets = np.append(0, np.cumsum([bases[key].ncoeffs for key in ['up','um','uz','p']]))
    pcoeff = evector[offsets[3]:offsets[4]]
    lengths, offsets = sph.coeff_sizes(Lmax, Nmax)
    pcoeff_full = np.zeros((Lmax,Nmax),dtype=pcoeff.dtype)
    for l in range(Lmax):
        pcoeff_full[l,:lengths[l]] = pcoeff[offsets[l]:offsets[l]+lengths[l]]

    projected_coeffs = normalize_coeffs(projected_coeffs)
    pcoeff_full = normalize_coeffs(pcoeff_full)
    coeff_error_m, coeff_error_p = projected_coeffs - pcoeff_full, projected_coeffs + pcoeff_full
    if np.max(abs(coeff_error_m)) > np.max(abs(coeff_error_p)):
        coeff_error = coeff_error_p
        pcoeff_full = -pcoeff_full
    else:
        coeff_error = coeff_error_m
    return coeff_error, projected_coeffs, pcoeff_full


def plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_fields, evalue_target=None):
    save_plots = True
    if m > 30:
        n = m+61
    else:
        n = 60
    num_modes = 6
    modes = list(zip([n]*num_modes, range(num_modes)))

    which_mode = 1
    modes = [modes[which_mode]]
    num_modes = len(modes)

    dense_solve = evalue_target is None

    # Load the data
    filename = pickle_filename(m, Lmax, Nmax, boundary_method, evalue_target=evalue_target)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    if save_plots:
        def save(fn, fig): save_figure(fn, fig)
    else:
        def save(fn, fig): pass

    configstr = f'm={m}-Lmax={Lmax}-Nmax={Nmax}-{boundary_method}'
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
    ns, neta = 1024, 512+1
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    Lmaxw, Nmaxw = vertical_velocity_size(Lmax, Nmax)
    bases = {'up': sph.Basis(s, eta, m, Lmax, Nmax, sigma=+1, alpha=1),
             'um': sph.Basis(s, eta, m, Lmax, Nmax, sigma=-1, alpha=1),
             'uz': sph.Basis(s, eta, m, Lmaxw, Nmaxw, sigma=0, alpha=1),
             'p':  sph.Basis(s, eta, m, Lmax, Nmax, sigma=0, alpha=0),
             'p2': sph.Basis(s, eta, m, Lmax, Nmax, sigma=0, alpha=2)}

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
    if num_modes == 1:
        ax = [ax]
    for i, (n, ell) in enumerate(modes):
        # Compute the analytic eigenfrequency
        mode_index = (n,(n-m)//2-ell,m)
        evalue_target = 2*greenspan.compute_eigenvalues(mode_index[0], mode_index[2])[mode_index[1]-1]
        index = np.argmin(abs(evalues - evalue_target))
        evalue, evector = evalues[index], evectors[:,index]
    
        print('Plotting eigenvector with eigenvalue {:1.8f}, resolution = {}'.format(evalue_target,(Lmax,Nmax)))
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

        # Compute the coefficient-space error
        coeff_error, projected_coeffs, pcoeff_full = compute_coeff_error(bases, mode_index, evector)

        coeff_fig, coeff_ax = plt.subplots()
        Nsizes = sph.Nsizes(Lmax, Nmax)
        for l in range(0,Lmax,2):
            c = coeff_error[l,:Nsizes[l]]
            coeff_ax.semilogy(abs(c))
        coeff_ax.grid(True)
        coeff_ax.set_xlabel('Radial Degree')
        coeff_ax.set_ylabel('Absolute Error')
        coeff_ax.set_title('Coefficient Error')
        dense = 'dense' if dense_solve else 'sparse'
        coeff_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        filename = prefix + '-coeff_error-' + configstr + f'-n={n}-ell={ell}-{dense}.png'
        save(filename, coeff_fig)

        print('    Projected coefficient l2 error: {:1.5e}'.format(np.linalg.norm((coeff_error).ravel())))

        p /= np.max(abs(p))
        if np.max(abs(panalytic-p)) > np.max(abs(panalytic+p)):
            p *= -1
    
        # Compute the analytic eigenfrequency
        error_evalue = evalue_target - evalue
        print(f'    Eigenvalue error: {abs(error_evalue):1.5e}')

        error = panalytic - p.real
        print(f'    Eigenfunction error: {np.max(abs(error)):1.5e}')

        # Spherical radial velocity component
        bc_error_top = np.max(np.abs(ur[-1,:]))
        bc_error_bot = np.max(np.abs(ur[ 0,:]))
        print(f'    Top    boundary error: {bc_error_top:1.3e}')
        print(f'    Bottom boundary error: {bc_error_bot:1.3e}')

        # Error plot
        plot_error = False
        if plot_error:
            sph.plotfield(s.ravel(), eta.ravel(), panalytic, colorbar=False)
            plt.title('analytic mode')
        
            sph.plotfield(s.ravel(), eta.ravel(), error)
            plt.title('error')

        # plot_spectrum_callback(index, evalues, evectors, m, Lmax, Nmax, s, eta, bases)

    suffix = 'dense' if evalue_target is None else 'sparse'
    filename = prefix + '-evectors-' + configstr + f'-n={n}-p-{suffix}.png'
    save(filename, fig)


def analyze_resolution(m):
    boundary_method = 'tau'

    if m == 30:
        Nmax, n = 24, 60
    else:
        Nmax, n = 32, m+61
    Lmax_values = list(range(8,42,2))

    num_modes = 6
    mode_targets = [(n,(n-m)//2-i,m) for i in range(num_modes)]
    evalue_targets = [2*greenspan.compute_eigenvalues(i[0], i[2])[i[1]-1] for i in mode_targets]

    errors = np.zeros((len(evalue_targets),len(Lmax_values)))
    for i,Lmax in enumerate(Lmax_values):
        for k,evalue_target in enumerate(evalue_targets):
            # Load the data
            filename = pickle_filename(m, Lmax, Nmax, boundary_method, evalue_target=evalue_target)
            data = pickle.load(open(filename, 'rb'))
            evalues, evectors = data['evalues'], data['evectors']

            # Compute the eigenvalue error
            index = np.argmin(abs(evalues - evalue_target))
            evalue, evector = evalues[index], evectors[:,index]
            error = evalue-evalue_target
            errors[k,i] = abs(error)

    fig, ax = plt.subplots()
    markers = ['o', 'v', 's', 'P', 'X', 'D']
    for k in range(num_modes):
        ax.semilogy(Lmax_values, errors[k].ravel(), label=f'λ = {evalue_targets[k]:1.4f}', marker=markers[k], markersize=8)
    ax.set_xlabel(r'$L_{\mathregular{max}}$')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'm = {m}')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    prefix = filename_prefix(directory='figures')
    filename = prefix + f'-evalue_error-vs-Lmax-m={m}-n={n}-Nmax={Nmax}.png'
    save_figure(filename, fig)



def _solve_helper(m, Lmax, Nmax, evalue_target):
    boundary_method = 'tau'

    # Skip if we already have it
    filename = pickle_filename(m, Lmax, Nmax, boundary_method, evalue_target=evalue_target)
    if os.path.exists(filename):
        print('  Already solved')
        return

    solve_eigenproblem(m, Lmax, Nmax, boundary_method, evalue_target=evalue_target, plot_spy=False)


def solve():
    m, num_modes = 95, 6
    Lmax_values = list(range(4,42,2))
    if m == 30:
        Nmax_values = [24]
        n = 60
    else:
        Nmax_values = [32]
        n = m+61
    mode_targets = [(n,(n-m)//2-i,m) for i in range(num_modes)]
    evalue_targets = [2*greenspan.compute_eigenvalues(i[0], i[2])[i[1]-1] for i in mode_targets]
    configs = itertools.product([m], Lmax_values, Nmax_values, evalue_targets)

    pool = mp.Pool(mp.cpu_count()-1)
    pool.starmap(_solve_helper, configs)


def plot_greenspan_modes_nostretch(m, n, num_modes):
    mode_targets = [(n,(n-m)//2-i,m) for i in range(num_modes)]

    s, eta = np.linspace(0,1,1024), np.linspace(-1,1,513)
    z = np.sqrt(1-s[np.newaxis,:]**2) * eta[:,np.newaxis]

    fig, ax = plt.subplots(1,num_modes, figsize=plt.figaspect(2/num_modes))
    for i, mode_target in enumerate(mode_targets):
        evalue = 2*greenspan.compute_eigenvalues(mode_target[0], mode_target[2])[mode_target[1]-1]
        mode = greenspan.compute_eigenmode(s, z, *mode_target, normalize=True)

        sph.plotfield(s, eta, mode, fig=fig, ax=ax[i], colorbar=False)
        ax[i].set_title(f'λ = {evalue:1.4f}')
        ax[i].set_xticks(np.linspace(0,1,3))
        ax[i].set_xlabel('s')
        if i > 0:
            ax[i].set_yticklabels([])
            ax[i].set_ylabel(None)

    prefix = filename_prefix(directory='figures')
    filename = prefix + f'-m={m}-n={n}-greenspan_solutions-nostretch.png'
    save_figure(filename, fig)


def plot_greenspan_modes(m, n, num_modes):
    mode_targets = [(n,(n-m)//2-i,m) for i in range(num_modes)]

    s, eta = np.linspace(0,1,1024), np.linspace(-1,1,513)
    z = np.sqrt(1-s[np.newaxis,:]**2) * eta[:,np.newaxis]

    fig, ax = plt.subplots(2,num_modes, figsize=(12.75,8))
    for i, mode_target in enumerate(mode_targets):
        evalue = 2*greenspan.compute_eigenvalues(mode_target[0], mode_target[2])[mode_target[1]-1]
        mode = greenspan.compute_eigenmode(s, z, *mode_target, normalize=True)

        sph.plotfield(s, eta, mode, fig=fig, ax=ax[0][i], colorbar=False)
        sph.plotfield(s, eta, mode, fig=fig, ax=ax[1][i], colorbar=False, stretch=True)
        ax[0][i].set_title(f'λ = {evalue:1.4f}')
        ax[0][i].set_xticklabels([])
        ax[0][i].set_xlabel(None)
        ax[1][i].set_xticks(np.linspace(0,1,3))
        if i > 0:
            ax[0][i].set_yticklabels([])
            ax[0][i].set_ylabel(None)
            ax[1][i].set_yticklabels([])
            ax[1][i].set_ylabel(None)

    prefix = filename_prefix(directory='figures')
    filename = prefix + f'-m={m}-n={n}-greenspan_solutions.png'
    save_figure(filename, fig)


def main():
    solve = True
    plot_evalues = False
    plot_fields = True
    plot_spy = False

    m = 30
    resolutions = [(16,16),(18,20),(18,24),(18,32),(32,32),(40,40)]
    boundary_method = 'tau'

    evalue_target = -0.13933425
#    evalue_target = None

    print(f'Inertial Waves, m = {m}')

    for Lmax,Nmax in resolutions:
        print(f'  Domain size: Lmax = {Lmax}, Nmax = {Nmax}')
        if solve:
            solve_eigenproblem(m, Lmax, Nmax, boundary_method, plot_spy, evalue_target=evalue_target)

        if plot_fields or plot_evalues:
            plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_fields, evalue_target=evalue_target)
    plt.show()


if __name__=='__main__':
#    main()
#    solve()
#    analyze_resolution(30)
#    analyze_resolution(95)

#    plot_greenspan_modes(30, 60, 6)
#    plot_greenspan_modes(95, 95+61, 6)

    plot_greenspan_modes_nostretch(30, 60, 6)
    plot_greenspan_modes_nostretch(95, 95+61, 6)
    plt.show()


