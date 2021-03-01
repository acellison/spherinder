import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import os
import pickle

import spherinder.operators as sph
from spherinder.eigtools import eigsort, scipy_sparse_eigs, plot_spectrum, discard_spurious_eigenvalues
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
    nvar = Lmax*Nmax
    neqn = (Lmax+2)*(Nmax+1)
    ntau = 2*(Nmax+1)+Lmax

    variables = [range(i*nvar,(i+1)*nvar) for i in range(nfields)]
    equations = [range(i*neqn,(i+1)*neqn) for i in range(nfields)]

    vartau = range(nfields*nvar,nfields*(nvar+ntau))
    varindices = [val for tup in zip(*variables) for val in tup]
    varindices = varindices + list(vartau)
    eqnindices = [val for tup in zip(*equations) for val in tup]
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


def output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, directory, ext, prefix='evalues'):
    truncstr = '-truncated'
    return make_filename_prefix(directory) + f'-{prefix}-m={m}-Lmax={Lmax}-Nmax={Nmax}-Ekman={Ekman:1.4e}-Prandtl={Prandtl}-Rayleigh={Rayleigh:1.4e}-{boundary_method}{truncstr}' + ext


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
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, directory='data', ext='.pckl')
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


def expand_evectors(vec, bases, fields='all', error_only=False):
    ncoeffs = [b.ncoeffs for b in [bases['up'], bases['um'], bases['uz'], bases['p'], bases['T']]]
    offsets = np.append(0, np.cumsum(ncoeffs))

    tau = vec[offsets[-1]:]
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


def plot_spectrum_callback(index, evalues, evectors, bases, error_only=False):
    evalue, evector = evalues[index], evectors[:,index]

    fields = ['p']
    d = expand_evectors(evector, bases, fields=fields, error_only=error_only)
    if error_only:
        return

    fig, ax = plt.subplots(1,len(fields),figsize=(0.5+3*len(fields),4.5))
    if len(fields) == 1:
        ax = [ax]
    s, eta = bases['up'].s, bases['up'].eta
    for i, field in enumerate(fields):
        f = d[field].real if np.linalg.norm(d[field].real) >= np.linalg.norm(d[field].imag) else d[field].imag
        sph.plotfield(s, eta, f, fig=fig, ax=ax[i], colorbar=len(fields)>1)
        ax[i].set_title(r'${}$'.format(field))

    fig.suptitle('Eigenvalue: {:1.4e}'.format(evalue))
    fig.set_tight_layout(True)

    fig.show()


def plot_solution(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, plot_evalues, plot_fields):
    # Load the data
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, directory='data', ext='.pckl')
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    # Now only plot the eigenvalues with accurate eigenvectors
    tolerance = np.inf
    if tolerance < np.inf:
        nfields = 5
        Nlengths, _ = sph.coeff_sizes(Lmax, Nmax)
        tauoffset = nfields * (np.sum(Nlengths) if boundary_method == 'galerkin' else Lmax*Nmax)
        good = [np.linalg.norm(evectors[tauoffset:,i]) < tolerance for i in range(len(evalues))]
        print('Number of eigenvectors with tau norm below {}: {}/{}'.format(tolerance, np.sum(good), len(evalues)))
        evalues, evectors = evalues[good], evectors[:,good]

    error_only = not plot_fields
    ns, neta = 1024, 1025
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    bases = create_bases(m, Lmax, Nmax, boundary_method, s, eta)
    onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, bases, error_only=error_only)

    fig, ax = plot_spectrum(evalues, onpick)
    ax.set_title('Boussinesq Eigenvalues')

    plot_filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, directory='figures', ext='.png')
    save_figure(plot_filename, fig)


def plot_gravest_modes(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh):
    # Load the data
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, directory='data', ext='.pckl')
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']
    nev = len(evalues)
    fieldname = 'u'

    # Create the plotting domain
    ns, neta = 256, 255
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    bases = create_bases(m, Lmax, Nmax, boundary_method, s, eta)

    # Plot
    nrows, ncols = 2, 5
    assert nev >= nrows*ncols

    fig, plot_axes = plt.subplots(nrows,ncols,figsize=(2*ncols,3*nrows))
    row, col = 0, 0
    for ii in range(nrows*ncols):
        evalue, evector = evalues[-1-ii], evectors[:,-1-ii]
        field = expand_evectors(evector, bases, fields=[fieldname])[fieldname]

        # Plot the field
        ax = plot_axes[row][col]
        sph.plotfield(s, eta, field.real, fig=fig, ax=ax, colorbar=False)

        # Adjust the axes labels
        if col != 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        if row != nrows-1:
            ax.set_xlabel('')
            ax.set_xticklabels([])

        # Bump row and column indices
        col += 1
        if col >= ncols:
            row += 1
            col = 0

    # Save the figure
    plot_filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, directory='figures', ext='.png', prefix='fields')
    save_figure(plot_filename, fig, dpi=600)


def rotation_configs():
    return [{'Ekman': 10**-4,   'm': 6,  'omega': -.43346, 'Rayleigh': 5.1549, 'Lmax': 10, 'Nmax': 20},
            {'Ekman': 10**-4.5, 'm': 9,  'omega': -.44276, 'Rayleigh': 4.7613, 'Lmax': 60, 'Nmax': 40},
            {'Ekman': 10**-5,   'm': 14, 'omega': -.45715, 'Rayleigh': 4.5351, 'Lmax': 40, 'Nmax': 40},
            {'Ekman': 10**-5.5, 'm': 20, 'omega': -.45760, 'Rayleigh': 4.3937, 'Lmax': 100, 'Nmax': 60},
            {'Ekman': 10**-6,   'm': 30, 'omega': -.46394, 'Rayleigh': 4.3021, 'Lmax': 120, 'Nmax': 80},
            {'Ekman': 10**-6.5, 'm': 44, 'omega': -.46574, 'Rayleigh': 4.2416, 'Lmax': 140, 'Nmax': 100},
            {'Ekman': 10**-7,   'm': 65, 'omega': -.46803, 'Rayleigh': 4.2012, 'Lmax': 160, 'Nmax': 120},
            {'Ekman': 10**-7.5, 'm': 95, 'omega': -.46828, 'Rayleigh': 4.1742, 'Lmax': 240, 'Nmax': 200},
            {'Ekman': 10**-8, 'm': 139, 'omega': -.43507, 'Rayleigh': 4.1527, 'Lmax': 120, 'Nmax': 200},
            {'Ekman': 10**-9, 'm': 300, 'omega': -.43507, 'Rayleigh': 4.1527, 'Lmax': 240, 'Nmax': 300},
            {'Ekman': 10**-10, 'm': 646, 'omega': -.43507, 'Rayleigh': 4.1527, 'Lmax': 450, 'Nmax': 450},
            {'Ekman': 10**-11, 'm': 1392, 'omega': -.43507, 'Rayleigh': 4.1527, 'Lmax': 580, 'Nmax': 400},
            ]


def _solve_helper(config_index):
    boundary_method = 'galerkin'
    boundary_condition = 'no-slip'
    nev = 10

    config = rotation_configs()[config_index]
    m, Lmax, Nmax = config['m'], config['Lmax'], config['Nmax']
    Ekman, Prandtl, Rayleigh = config['Ekman'], 1, config['Rayleigh']

    # Skip if we already have it
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, directory='data', ext='.pckl')
    if os.path.exists(filename):
        print('  Already solved')
        return

    omega = config['omega']
    solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, omega, plot_spy, nev=nev)


def solve():
    indices = range(8)

    pool = mp.Pool(mp.cpu_count()-1)
    pool.map(_solve_helper, indices)


def main():
    solve = True
    plot_spy = False
    plot_evalues = True
    plot_fields = True
    boundary_method = 'galerkin'
    nev = 'all'
#    nev = 10

    config_index = 0
    config = rotation_configs()[config_index]

    m, Ekman, Prandtl, Rayleigh, omega = config['m'], config['Ekman'], 1, config['Rayleigh'], config['omega']
    Lmax, Nmax = config['Lmax'], config['Nmax']

    omega = 1j*omega/Ekman**(2/3)
    Rayleigh /= Ekman**(1/3)

    print(f'Linear onset, m = {m}, Ekman = {Ekman:1.4e}, Prandtl = {Prandtl}, Rayleigh = {Rayleigh:1.4e}')
    print('  Domain size: Lmax = {}, Nmax = {}'.format(Lmax, Nmax))
    print('  Boundary method = ' + boundary_method)

    if solve:
        solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, omega, plot_spy, nev=nev)

    if plot_fields or plot_evalues:
        plot_solution(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, plot_evalues, plot_fields)
        plt.show()


def analyze_evalues():
    config_index = 4
    config = rotation_configs()[config_index]

    m, Ekman, Prandtl, Rayleigh, omega = config['m'], config['Ekman'], 1, config['Rayleigh'], config['omega']
    boundary_method = 'galerkin'

    omega = 1j*omega/Ekman**(2/3)
    Rayleigh /= Ekman**(1/3)

    Llores, Nlores = 48, 48
    Lhires, Nhires = 52, 52

    def load_evalues(L, N):
        filename = output_filename(m, L, N, boundary_method, Ekman, Prandtl, Rayleigh, directory='data', ext='.pckl')
        data = pickle.load(open(filename, 'rb'))
        return data['evalues'], data['evectors']
        
    evalues_lores, evectors_lores = load_evalues(Llores, Nlores)
    evalues_hires, _ = load_evalues(Lhires, Nhires)
    evalues, indices = discard_spurious_eigenvalues(evalues_lores, evalues_hires, cutoff=1e4, plot=True)
    evectors = evectors_lores[:,indices]

    print('Number of good eigenvalues: {}/{}'.format(len(evalues), len(evalues_lores)))

    if True:
        Lmax, Nmax = Llores, Nlores
        ns, neta = 256, 255
        s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
        bases = create_bases(m, Lmax, Nmax, boundary_method, s, eta)
        onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, Lmax, Nmax, s, eta, bases)
    else:
        onpick = None

    fig, ax = plot_spectrum(evalues, onpick)
    ax.set_title('Resolved Boussinesq Eigenvalues')
    plt.show()


if __name__=='__main__':
    main()
#    analyze_evalues()

