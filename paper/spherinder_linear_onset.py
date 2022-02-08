import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import os
import pickle
import multiprocessing as mp

import spherinder.operators as sph
from spherinder.eigtools import eigsort, scipy_sparse_eigs, plot_spectrum
from fileio import save_data, save_figure


g_alpha_T = 0
g_file_prefix = 'spherinder_linear_onset'


def matrices_galerkin(m, Lmax, Nmax, Ekman, Prandtl, Rayleigh):
    """Construct matrices for X = [u(+), u(-), u(z), p, T]
    """
    alpha_bc, alpha_bc_s, alpha_bc_T = 2, 1, g_alpha_T+1

    Lout, Nout = Lmax+2, Nmax+1
    Loutw, Noutw = Lout, Nout
    ncoeff = sph.num_coeffs(Lout, Nout)
    ncoeff0 = sph.num_coeffs(Lmax, Nmax)

    # Galerkin conversion operators
    Boundp = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=+1, exact=True)
    Boundm = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=-1, exact=True)
    Boundz = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=0,  exact=True)
    BoundT = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=1+g_alpha_T, sigma=0, exact=True)

    # Scalar laplacian
    LapT = sph.operator('lap')(m, Lout, Nout, alpha=g_alpha_T)

    # Vector laplacian
    Lapp, Lapm, Lapz = sph.operator('lap', 'vec')(m, Lout, Nout, alpha=1)

    # Vector divergence operator
    Div = sph.operator('div')(m, Lout, Nout, alpha=1)
    Divp, Divm, Divz = Div[:,:ncoeff], Div[:,ncoeff:2*ncoeff], Div[:,2*ncoeff:]

    # Pressure gradient
    Gradp, Gradm, Gradz = sph.operator('grad')(m, Lmax, Nmax, alpha=2)

    # Pad the pressure gradient to fit the Galerkin coefficient sizes
    Gradp = sph.resize(Gradp, Lmax, Nmax, Lout, Nout)
    Gradm = sph.resize(Gradm, Lmax, Nmax, Lout, Nout)
    Gradz = sph.resize(Gradz, Lmax, Nmax, Lout, Nout)

    # Radial vector extraction
    Rad = sph.operator('rdot')(m, Lout, Nout, alpha=1, exact=False)
    Cr = sph.convert_alpha(1+g_alpha_T, m, Lout, Nout, alpha=1, sigma=0)
    Rad = Cr @ Rad
    Radp, Radm, Radz = Rad[:,:ncoeff], Rad[:,ncoeff:2*ncoeff], Rad[:,2*ncoeff:]

    # Radial vector multiplication r e_r * T, convert from alpha=g_alpha_T to alpha=3
    RTp, RTm, RTz = sph.operator('rtimes')(m, Lout, Nout, alpha=g_alpha_T, exact=False)
    CrTp = sph.convert_alpha(3-g_alpha_T, m, Lout, Nout, alpha=g_alpha_T, sigma=+1)
    CrTm = sph.convert_alpha(3-g_alpha_T, m, Lout, Nout, alpha=g_alpha_T, sigma=-1)
    CrTz = sph.convert_alpha(3-g_alpha_T, m, Lout, Nout, alpha=g_alpha_T, sigma=0)
    RTp, RTm, RTz = CrTp @ RTp, CrTm @ RTm, CrTz @ RTz

    # Conversion matrices
    Cp = sph.convert_alpha(2, m, Lout, Nout, alpha=1, sigma=+1)
    Cm = sph.convert_alpha(2, m, Lout, Nout, alpha=1, sigma=-1)
    Cz = sph.convert_alpha(2, m, Lout, Nout, alpha=1, sigma=0)
    CT = sph.convert_alpha(2, m, Lout, Nout, alpha=g_alpha_T, sigma=0)
    
    # Time derivative matrices
    M00 = Ekman * Cp @ Boundp
    M11 = Ekman * Cm @ Boundm
    M22 = Ekman * Cz @ Boundz
    M33 = sparse.lil_matrix((ncoeff,ncoeff0))
    M44 = Prandtl * CT @ BoundT

    # i*u+ equation - spin+ velocity component
    L00 = (-1j * Cp + Ekman * Lapp) @ Boundp
    L01 = sparse.lil_matrix((ncoeff,ncoeff0))
    L02 = sparse.lil_matrix((ncoeff,ncoeff0))
    L03 = -Gradp
    L04 = Rayleigh * RTp @ BoundT

    # i*u- equation - spin- velocity component
    L10 = sparse.lil_matrix((ncoeff,ncoeff0))
    L11 = (1j * Cm + Ekman * Lapm) @ Boundm
    L12 = sparse.lil_matrix((ncoeff,ncoeff0))
    L13 = -Gradm
    L14 = Rayleigh * RTm @ BoundT

    # i*w equation - vertical velocity component
    L20 = sparse.lil_matrix((ncoeff,ncoeff0))
    L21 = sparse.lil_matrix((ncoeff,ncoeff0))
    L22 = Ekman * Lapz @ Boundz
    L23 = -Gradz
    L24 = Rayleigh * RTz @ BoundT

    # Divergence equation
    L30 = Divp @ Boundp
    L31 = Divm @ Boundm
    L32 = Divz @ Boundz
    L33 = sparse.lil_matrix((ncoeff,ncoeff0))
    L34 = sparse.lil_matrix((ncoeff,ncoeff0))

    # Temperature equation
    L40 = Prandtl * Radp @ Boundp
    L41 = Prandtl * Radm @ Boundm
    L42 = Prandtl * Radz @ Boundz
    L43 = sparse.lil_matrix((ncoeff,ncoeff0))
    L44 = LapT @ BoundT

    Mmats = [M00, M11, M22, M33, M44]
    upmats = [L00, L01, L02, L03, L04]
    ummats = [L10, L11, L12, L13, L14]
    uzmats = [L20, L21, L22, L23, L24]
    pmats = [L30, L31, L32, L33, L34]
    Tmats = [L40, L41, L42, L43, L44]

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([upmats, ummats, uzmats, pmats, Tmats], format=sparse_format)

    # Tau polynomials
    def tau_polynomials():
        Taup = sph.tau_projection(m, Lout,  Nout,  alpha=3, sigma=+1, alpha_bc=alpha_bc)
        Taum = sph.tau_projection(m, Lout,  Nout,  alpha=3, sigma=-1, alpha_bc=alpha_bc)
        Tauz = sph.tau_projection(m, Loutw, Noutw, alpha=3, sigma=0,  alpha_bc=alpha_bc)
        Taus = sph.tau_projection(m, Lout,  Nout,  alpha=2, sigma=0,  alpha_bc=alpha_bc_s)
        Taut = sph.tau_projection(m, Lout, Nout, alpha=2+g_alpha_T, sigma=0, alpha_bc=alpha_bc_T)

        return sparse.block_diag([Taup, Taum, Tauz, Taus, Taut])

    col = tau_polynomials()

    L = sparse.hstack([L,  col], format='csr')
    M = sparse.hstack([M,0*col], format='csr')

    return M, L


def matrices(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh):
    if boundary_method in ['galerkin']:
        return eval('matrices_' + boundary_method)(m, Lmax, Nmax, Ekman, Prandtl, Rayleigh)
    else:
        raise ValueError('Unsupported boundary method')


def permutation_indices(Lmax, Nmax):
    """For each mode interlace the five field variables.  Returns two lists of
       permutation indices, the first for the columns (variable ordering), and
       the second for the rows (equation sorting).  Leaves tau variables as the
       final set of coefficients so the tau columns are in the same location -
       horizontally block appended to the matrix"""
    nfields = 5
    nvar = sph.num_coeffs(Lmax, Nmax)
    neqn = sph.num_coeffs(Lmax+2, Nmax+1)
    ntau = neqn - nvar

    lengths, offsets = sph.coeff_sizes(Lmax, Nmax)
    varindices = []
    for ell in range(Lmax):
        offset, length = offsets[ell], lengths[ell]
        variables = [list(range(offset+i*nvar, offset+i*nvar+length)) for i in range(nfields)]
        varindices += np.ravel(variables).tolist()
    vartau = range(nfields*nvar,nfields*neqn)
    varindices = varindices + list(vartau)

    lengths, offsets = sph.coeff_sizes(Lmax+2, Nmax+1)
    eqnindices = []
    for ell in range(Lmax+2):
        offset, length = offsets[ell], lengths[ell]
        equations = [list(range(offset+i*neqn, offset+i*neqn+length)) for i in range(nfields)]
        eqnindices += np.ravel(equations).tolist()

    return varindices, eqnindices


def invert_permutation(permutation):
    """Invert a permutation"""
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv


def make_filename_prefix(directory='data'):
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), directory))
    abspath = os.path.join(basepath, g_file_prefix)
    return os.path.join(abspath, g_file_prefix)


def output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, nev, directory, ext, prefix='evalues'):
    nevstr = f'-nev={nev}' if isinstance(nev, int) else ''
    return make_filename_prefix(directory) + f'-{prefix}-m={m}-Lmax={Lmax}-Nmax={Nmax}-Ekman={Ekman:1.4e}-Prandtl={Prandtl}-Rayleigh={Rayleigh:1.4e}-{boundary_method}{nevstr}' + ext


def solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, omega, plot_spy, nev='all'):
    # Construct the system
    print('Constructing matrix system...')
    M, L = matrices(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh)

    permute = False  # Fixme: implement permutation for triangular truncation
    enable_permutation = permute and boundary_method == 'galerkin'
    if enable_permutation:
        print('Reordering variables and equations...')
        var, eqn = permutation_indices(Lmax, Nmax)
        M, L = M[:,var], L[:,var]
        M, L = M[eqn,:], L[eqn,:]

    if plot_spy:
        fig, plot_axes = plt.subplots(1,2,figsize=(9,4))
        plot_axes[0].spy(L)
        plot_axes[1].spy(M)
        plt.show()

    # Compute the eigenvalues and eigenvectors
    print('Eigenvalue problem, size {}'.format(np.shape(L)))
    if nev == 'all':
        evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)
    else:
        matsolver = 'UmfpackFactorized64'
        evalues, evectors = scipy_sparse_eigs(L, M, nev, target=omega, matsolver=matsolver, profile=True)

    if enable_permutation:
        vari = invert_permutation(var)
        evectors = evectors[vari,:]

    # Output data
    data = {'m': m, 'Lmax': Lmax, 'Nmax': Nmax, 
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors,
            'Ekman': Ekman, 'Prandtl': Prandtl, 'Rayleigh': Rayleigh}
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, nev=nev, directory='data', ext='.pckl')
    save_data(filename, data)


def create_bases(m, Lmax, Nmax, boundary_method, s, eta):
    if boundary_method == 'galerkin':
        galerkin = True
        dalpha = 1
    else:
        galerkin = False
        dalpha = 0

    upbasis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=+1, alpha=1+dalpha, galerkin=galerkin)
    umbasis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=-1, alpha=1+dalpha, galerkin=galerkin)
    uzbasis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=0,  alpha=1+dalpha, galerkin=galerkin)
    pbasis  = sph.Basis(s, eta, m, Lmax, Nmax, sigma=0,  alpha=2, galerkin=False)
    Tbasis  = sph.Basis(s, eta, m, Lmax, Nmax, sigma=0,  alpha=g_alpha_T+dalpha, galerkin=galerkin)
    bases = {'up':upbasis, 'um':umbasis, 'uz':uzbasis, 'p':pbasis, 'T':Tbasis}

    return bases


def create_equatorial_bases(m, Lmax, Nmax, boundary_method, ns, nphi):
    s, eta = np.linspace(0,1,ns+1)[1:], np.array([0.])
    bases = create_bases(m, Lmax, Nmax, boundary_method, s, eta)

    s, phi = s[np.newaxis,:], np.linspace(0,2*np.pi,nphi+1)[:,np.newaxis]
    x, y, mode = s*np.cos(phi), s*np.sin(phi), np.exp(1j*m*phi)

    return bases, x, y, phi, mode


def expand_evectors(vec, bases, fields='all', error_only=False, verbose=True):
    ncoeffs = [b.ncoeffs for b in [bases['up'], bases['um'], bases['uz'], bases['p'], bases['T']]]
    offsets = np.append(0, np.cumsum(ncoeffs))

    tau = vec[offsets[-1]:]
    if verbose:
        print('Tau norm: {}'.format(np.linalg.norm(tau)))
    if error_only:
        return

    # Get the grid space vector fields
    nfields = 5
    upcoeff, umcoeff, uzcoeff, pcoeff, Tcoeff = tuple(vec[offsets[i]:offsets[i+1]] for i in range(nfields))
    coeffs = {'up':upcoeff, 'um':umcoeff, 'uz':uzcoeff, 'p':pcoeff, 'T':Tcoeff}
    if fields == 'all':
        fields = ['u','v','w','p','T']

    which = []
    if 'u' in fields or 'v' in fields: which += ['up', 'um']
    if 'w' in fields: which += ['uz']
    if 'p' in fields: which += ['p']
    if 'T' in fields: which += ['T']

    # Convert to grid space
    result = [bases[field].expand(coeffs[field]) for field in which]

    # Collect results
    get_field = lambda field: result[which.index(field)] if field in which else None
    up, um = get_field('up'), get_field('um')
    if up is not None and um is not None:
        u, v = np.sqrt(0.5)*(up + um), -1j * np.sqrt(0.5)*(up - um)
    else:
        u, v = None, None
    w, p, T = get_field('uz'), get_field('p'), get_field('T')

    return {'u':u, 'v':v, 'w':w, 'p':p, 'T':T}


def plot_spectrum_callback(index, evalues, evectors, bases, equatorial_bases=None, error_only=False, fig=None, ax=None):
    evalue, evector = evalues[index], evectors[:,index]

    fields = ['p']
    d = expand_evectors(evector, bases, fields=fields, error_only=error_only)
    if error_only:
        return

    if fig is None or ax is None:
        fig, ax = plt.subplots(1,len(fields),figsize=(0.5+3*len(fields),4.5))
    if len(fields) == 1: ax = [ax]

    s, eta = bases['up'].s, bases['up'].eta
    for i, field in enumerate(fields):
        f = d[field].real if np.linalg.norm(d[field].real) >= np.linalg.norm(d[field].imag) else d[field].imag
        sph.plotfield(s, eta, f, fig=fig, ax=ax[i], colorbar=len(fields)>1)
        ax[i].set_title(r'${}$'.format(field))

    fig.suptitle('λ = {:1.4e}'.format(evalue))
    fig.set_tight_layout(True)
    fig.show()

    if equatorial_bases is None:
        return

    bases, x, y, phi, mode = equatorial_bases
    d = expand_evectors(evector, bases, fields=fields, error_only=error_only)

    fig, ax = plt.subplots(1,len(fields),figsize=(0.5+5*len(fields),4.5))
    if len(fields) == 1: ax = [ax]

    for i, field in enumerate(fields):
        f = mode * d[field]
        f = f.real if np.linalg.norm(f.real) >= np.linalg.norm(f.imag) else f.imag

        im = ax[i].pcolormesh(x, y, f, cmap='RdBu', shading='gouraud')
        ax[i].plot(np.cos(phi), np.sin(phi), color='k', linewidth=0.5, alpha=0.5)
        ax[i].set_aspect(aspect='equal', adjustable='datalim')
        ax[i].set_title(r'${}$'.format(field))

        if i > 0:
            ax[i].set_yticklabels([])
            ax[i].set_ylabel(None)

    fig.suptitle('λ = {:1.4f}'.format(evalue))
    fig.set_tight_layout(True)
    fig.show()


def plot_solution(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, plot_evalues, plot_fields, nev='all'):
    # Load the data
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, nev=nev, directory='data', ext='.pckl')
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    # Now only plot the eigenvalues with accurate eigenvectors
    tolerance = np.inf
    if tolerance < np.inf:
        nfields = 5
        tauoffset = nfields * sph.num_coeffs(Lmax, Nmax)
        good = [np.linalg.norm(evectors[tauoffset:,i]) < tolerance for i in range(len(evalues))]
        print('Number of eigenvectors with tau norm below {}: {}/{}'.format(tolerance, np.sum(good), len(evalues)))
        evalues, evectors = evalues[good], evectors[:,good]

    error_only = not plot_fields
    ns, neta = 1024, 1025
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    bases = create_bases(m, Lmax, Nmax, boundary_method, s, eta)
    equatorial_bases = create_equatorial_bases(m, Lmax, Nmax, boundary_method, ns=512, nphi=512)
    onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, bases, equatorial_bases=equatorial_bases, error_only=error_only)

    fig, ax = plot_spectrum(evalues, onpick)
    ax.set_title('Boussinesq Eigenvalues')

    plot_filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, nev=nev, directory='figures', ext='.png')
    save_figure(plot_filename, fig)


def rotation_configs():
    return [{'Ekman': 10**-4,   'm': 6,    'omega': -.27009, 'Rayleigh': 5.0151,  'Lmax': 100, 'Nmax': 100},
            {'Ekman': 10**-4.5, 'm': 9,    'omega': -.31035, 'Rayleigh': 4.6581,  'Lmax': 100, 'Nmax': 100},
            {'Ekman': 10**-5,   'm': 13,   'omega': -.33901, 'Rayleigh': 4.4660,  'Lmax': 100, 'Nmax': 100},
            {'Ekman': 10**-5.5, 'm': 20,   'omega': -.36778, 'Rayleigh': 4.3488,  'Lmax': 120, 'Nmax': 120},
            {'Ekman': 10**-6,   'm': 30,   'omega': -.38930, 'Rayleigh': 4.2736,  'Lmax': 160, 'Nmax': 160},
            {'Ekman': 10**-6.5, 'm': 44,   'omega': -.40439, 'Rayleigh': 4.22355, 'Lmax': 180, 'Nmax': 180},
            {'Ekman': 10**-7,   'm': 65,   'omega': -.41737, 'Rayleigh': 4.19025, 'Lmax': 200, 'Nmax': 200},
            {'Ekman': 10**-7.5, 'm': 95,   'omega': -.42658, 'Rayleigh': 4.16773, 'Lmax': 300, 'Nmax': 240},
            ]



def _solve_helper(config_index, nev):
    boundary_method = 'galerkin'
    Prandtl = 1

    config = rotation_configs()[config_index]
    m, Lmax, Nmax, Ekman, Rayleigh, omega = [config[a] for a in ['m', 'Lmax', 'Nmax', 'Ekman', 'Rayleigh', 'omega']]

    omega = 1j*omega/Ekman**(2/3)
    Rayleigh /= Ekman**(1/3)

    # Skip if we already have it
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, nev=nev, directory='data', ext='.pckl')
    if os.path.exists(filename):
        print('  Already solved')
        return

    solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, omega, plot_spy=False, nev=nev)


def solve(nev):
    indices = range(8)
    indices = [(index, nev) for index in indices]

    pool = mp.Pool(mp.cpu_count()-1)
    pool.starmap(_solve_helper, indices)


def plot_critical_modes():
    configs = rotation_configs()
    Prandtl, boundary_method = 1, 'galerkin'
    nev = 3
    solve(nev)

    nrows, ncols = 2, 4
    figsize = plt.figaspect(nrows/ncols)
    scale = 1.5
    figsize = scale*figsize[0]*0.65, scale*figsize[1]
    fig, plot_axes = plt.subplots(nrows, ncols, figsize=figsize)

    figsize = plt.figaspect(nrows/ncols)
    scale = 1.5
    figsize = scale*figsize[0]*0.95, scale*figsize[1]
    fig_eq, plot_axes_eq = plt.subplots(nrows, ncols, figsize=figsize)

    ns, neta = 256, 255
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)

    t, t_eq = np.linspace(0,np.pi,100), np.linspace(0,2*np.pi,200)
    radius = 0.35
    circle_x, circle_y = radius*np.sin(t), -radius*np.cos(t)
    circle_x_eq, circle_y_eq = radius*np.sin(t_eq), -radius*np.cos(t_eq)

    field = 'T'
    nplots = nrows*ncols
    if nplots == 1:
        configs = [configs[7]]
        plot_axes = [[plot_axes]]
        plot_axes_eq = [[plot_axes_eq]]
    else:
        configs = configs[:nrows*ncols]

    row, col = 0, 0
    for config in configs:
        ax = plot_axes[row][col]

        m, Lmax, Nmax, Ekman, Rayleigh = [config[a] for a in ['m', 'Lmax', 'Nmax', 'Ekman', 'Rayleigh']]
        bases = create_bases(m, Lmax, Nmax, boundary_method, s, eta)

        Rayleigh /= Ekman**(1/3)

        filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, nev=nev, directory='data', ext='.pckl')
        data = pickle.load(open(filename, 'rb'))

        evalues, evectors = [data[a] for a in ['evalues', 'evectors']]
        evalue, evector = evalues[-1], evectors[:,-1]

        d = expand_evectors(evector, bases, fields=[field], error_only=False, verbose=False)
        f = d[field].real if np.linalg.norm(d[field].real) >= np.linalg.norm(d[field].imag) else d[field].imag

        fmin, fmax = f.min(), f.max()
        fcenter = f[neta//2,0]
        if abs(fcenter-fmax) < abs(fcenter-fmin):
            f = -f
        cmap = 'RdBu_r' if field == 'T' else 'RdBu'
        sph.plotfield(s, eta, f, fig=fig, ax=ax, cmap=cmap, colorbar=False)

        if nplots != 1:
            ax.plot(circle_x, circle_y, '--k', alpha=0.8, linewidth=1)

        ax.set_yticks(np.linspace(-1,1,5))
        if col > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        ax.set_xticks(np.linspace(0,1,3))
        if row < nrows-1:
            ax.set_xlabel('')
            ax.set_xticklabels([])

        logek = np.log10(Ekman)
        if nplots == 1:
            title = 'Meridional Slice'
            title_eq = 'Equatorial Slice'
        else:
            title = 'E = $10^{' + f'{logek:1.1f}' + '}$'
            title_eq = title
        ax.set_title(title)

        # Plot the equatorial plane
        bases_eq, x_eq, y_eq, phi_eq, mode_eq =  create_equatorial_bases(m, Lmax, Nmax, boundary_method, ns=1024, nphi=1024)
        d = expand_evectors(evector, bases_eq, fields=[field], error_only=False, verbose=False)
        f = mode_eq * d[field]
        f = f.real if np.linalg.norm(f.real) >= np.linalg.norm(f.imag) else f.imag

        ax = plot_axes_eq[row][col]
        im = ax.pcolormesh(x_eq, y_eq, f, cmap='RdBu', shading='gouraud')
        ax.plot(np.cos(phi_eq), np.sin(phi_eq), color='k', linewidth=0.5, alpha=0.5)
        ax.set_aspect(aspect='equal', adjustable='datalim')
        ax.set_title(title_eq)
        if nplots != 1:
            ax.plot(circle_x_eq, circle_y_eq, '--k', alpha=0.8, linewidth=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_yticks(np.linspace(-1,1,5))
        if col > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        ax.set_xticks(np.linspace(-1,1,5))
        if row < nrows-1:
            ax.set_xlabel('')
            ax.set_xticklabels([])

        # Update the plot indices
        col += 1
        if col == ncols:
            col = 0
            row += 1

    if nplots == 1:
        name = f'-field={field}'
    else:
        name = f's-field={field}'
    fig.set_tight_layout(True)
    filename = make_filename_prefix('figures') + f'-critical_mode{name}.png'
    save_figure(filename, fig)

    fig_eq.set_tight_layout(True)
    filename = make_filename_prefix('figures') + f'-critical_mode{name}-equatorial.png'
    save_figure(filename, fig_eq)

    plt.show()


def main():
    solve = False
    plot_spy = False
    plot_evalues = True
    plot_fields = True
    boundary_method = 'galerkin'
#    nev = 'all'
    nev = 10 
    resolution_scale = 1.0

    config_index = 2
    config = rotation_configs()[config_index]

    m, Ekman, Prandtl, Rayleigh, omega = config['m'], config['Ekman'], 1, config['Rayleigh'], config['omega']
    Lmax, Nmax = config['Lmax'], config['Nmax']

    # Rescale the resolution
    Lmax, Nmax = [int(resolution_scale*a) for a in [Lmax,Nmax]]

    omega = 1j*omega/Ekman**(2/3)
    Rayleigh /= Ekman**(1/3)


    print(f'Linear onset, m = {m}, Ekman = {Ekman:1.4e}, Prandtl = {Prandtl}, Rayleigh = {Rayleigh:1.4e}, omega = {omega.imag:1.4e}')
    print('  Domain size: Lmax = {}, Nmax = {}'.format(Lmax, Nmax))
    print('  Boundary method = ' + boundary_method)

    if solve:
        solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, omega, plot_spy, nev=nev)

    if plot_fields or plot_evalues:
        plot_solution(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, plot_evalues, plot_fields, nev=nev)
        plt.show()


if __name__=='__main__':
#    main()
#    solve()
    plot_critical_modes()

