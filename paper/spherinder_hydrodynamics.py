import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import os
import pickle

#from spherinder import config
#config.default_normalize = False

import spherinder.operators as sph
from spherinder.eigtools import eigsort, plot_spectrum, scipy_sparse_eigs
from fileio import save_data, save_figure

import spherinder.asymptotic_operators as sao


g_file_prefix = 'spherinder_hydrodynamics'


use_full_vertical_velocity = False

def vertical_velocity_size(Lmax, Nmax, full=use_full_vertical_velocity):
    return (Lmax, Nmax) if full else (Lmax-1, Nmax)


def matrices_galerkin(m, Lmax, Nmax, Ekman):
    """Construct matrices for X = [u(+), u(-), u(z), p]
    """
    alpha_bc, alpha_bc_s = 2, 1

    Lout, Nout = Lmax+2, Nmax+1
    Lw, Nw = vertical_velocity_size(Lmax, Nmax)
    Loutw, Noutw = Lw+2, Nw+1
    ncoeff = sph.num_coeffs(Lout, Nout)
    ncoeff0 = sph.num_coeffs(Lmax, Nmax)
    ncoeffw = sph.num_coeffs(Lw, Nw)
    ncoutw = sph.num_coeffs(Loutw, Noutw)
    
    # Galerkin conversion operators
    Boundp = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=+1, exact=True)
    Boundm = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=-1, exact=True)
    Boundz = sph.operator('1-r**2')(m, Lw,   Nw,   alpha=2, sigma=0,  exact=True)

    # Vector laplacian
    Lapp, Lapm, Lapz = sph.operator('lap', 'vec')(m, Lout, Nout, alpha=1)

    # Vector divergence operator
    Div = sph.operator('div')(m, Lout, Nout, alpha=1)
    Divp, Divm, Divz = Div[:,:ncoeff], Div[:,ncoeff:2*ncoeff], Div[:,2*ncoeff:]

    if not use_full_vertical_velocity:
        _, Noffsets = sph.coeff_sizes(Lout, Nout)
        Lapz = Lapz[:Noffsets[-1],:Noffsets[-1]]
        Divz = Divz[:,:Noffsets[-1]]

    # Pressure gradient
    Gradp, Gradm, Gradz = sph.operator('grad')(m, Lmax, Nmax, alpha=2)

    # Pad the pressure gradient to fit the Galerkin coefficient sizes
    Gradp = sph.resize(Gradp, Lmax, Nmax, Lout,  Nout)
    Gradm = sph.resize(Gradm, Lmax, Nmax, Lout,  Nout)
    Gradz = sph.resize(Gradz, Lmax, Nmax, Loutw, Noutw)

    # Conversion matrices
    Cp = sph.convert_alpha(2, m, Lout,  Nout,  alpha=1, sigma=+1,)
    Cm = sph.convert_alpha(2, m, Lout,  Nout,  alpha=1, sigma=-1,)
    Cz = sph.convert_alpha(2, m, Loutw, Noutw, alpha=1, sigma=0)
    
    # Time derivative matrices
    M00 = Cp @ Boundp
    M11 = Cm @ Boundm
    M22 = Cz @ Boundz
    M33 = sparse.lil_matrix((ncoeff,ncoeff0))

    # i*u+ equation - spin+ velocity component
    L00 = (-2j * Cp + Ekman * Lapp) @ Boundp
    L01 = sparse.lil_matrix((ncoeff,ncoeff0))
    L02 = sparse.lil_matrix((ncoeff,ncoeffw))
    L03 = -Gradp

    # i*u- equation - spin- velocity component
    L10 = sparse.lil_matrix((ncoeff,ncoeff0))
    L11 = (2j * Cm + Ekman * Lapm) @ Boundm
    L12 = sparse.lil_matrix((ncoeff,ncoeffw))
    L13 = -Gradm

    # i*w equation - vertical velocity component
    L20 = sparse.lil_matrix((ncoutw,ncoeff0))
    L21 = sparse.lil_matrix((ncoutw,ncoeff0))
    L22 = Ekman * Lapz @ Boundz
    L23 = -Gradz

    # Divergence equation
    L30 = Divp @ Boundp
    L31 = Divm @ Boundm
    L32 = Divz @ Boundz
    L33 = sparse.lil_matrix((ncoeff,ncoeff0))

    Mmats = [M00, M11, M22, M33]
    upmats = [L00, L01, L02, L03]
    ummats = [L10, L11, L12, L13]
    uzmats = [L20, L21, L22, L23]
    pmats = [L30, L31, L32, L33]

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([upmats, ummats, uzmats, pmats], format=sparse_format)

    # Tau polynomials
    def tau_polynomials():
        Taup = sph.tau_projection(m, Lout,  Nout,  alpha=3, sigma=+1, alpha_bc=alpha_bc)
        Taum = sph.tau_projection(m, Lout,  Nout,  alpha=3, sigma=-1, alpha_bc=alpha_bc)
        Tauz = sph.tau_projection(m, Loutw, Noutw, alpha=3, sigma=0,  alpha_bc=alpha_bc)
        Taus = sph.tau_projection(m, Lout,  Nout,  alpha=2, sigma=0,  alpha_bc=alpha_bc_s)
        return sparse.block_diag([Taup, Taum, Tauz, Taus])

    col = tau_polynomials()

    L = sparse.hstack([L,  col], format='csr')
    M = sparse.hstack([M,0*col], format='csr')

    return M, L


def matrices(m, Lmax, Nmax, boundary_method, Ekman):
    if boundary_method in ['galerkin']:
        return eval('matrices_' + boundary_method)(m, Lmax, Nmax, Ekman)
    else:
        raise ValueError('Unsupported boundary method')


def permutation_indices(Lmax, Nmax):
    """For each mode interlace the five field variables.  Returns two lists of 
       permutation indices, the first for the columns (variable ordering), and
       the second for the rows (equation sorting).  Leaves tau variables as the
       final set of coefficients so the tau columns are in the same location -
       horizontally block appended to the matrix"""
    nfields = 4
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


def output_filename(m, Lmax, Nmax, boundary_method, Ekman, directory, ext, prefix='evalues', nev=None):
    nevstr = f'-nev={nev}' if nev else ''
    return make_filename_prefix(directory) + f'-{prefix}-m={m}-Lmax={Lmax}-Nmax={Nmax}-Ekman={Ekman:1.4e}-{boundary_method}{nevstr}' + ext


def solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, plot_spy, nev, evalue_target):
    # Construct the system
    print('Constructing matrix system...')
    M, L = matrices(m, Lmax, Nmax, boundary_method, Ekman)

    permute = False  # FIXME: sort out permutation for truncated series
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
        evalues, evectors = scipy_sparse_eigs(L, M, N=nev, target=evalue_target, matsolver=matsolver, profile=True)

    if enable_permutation:
        vari = invert_permutation(var)
        evectors = evectors[vari,:]

    # Output data
    data = {'m': m, 'Lmax': Lmax, 'Nmax': Nmax, 
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors,
            'Ekman': Ekman}
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, directory='data', ext='.pckl', nev=nev)
    save_data(filename, data)


def create_bases(m, Lmax, Nmax, boundary_method, s, eta):
    if boundary_method == 'galerkin':
        galerkin = True
        dalpha = 1
        Lw, Nw = vertical_velocity_size(Lmax, Nmax)
    else:
        raise ValueError('Boundary method not implemented')

    upbasis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=+1, alpha=1+dalpha, galerkin=galerkin)
    umbasis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=-1, alpha=1+dalpha, galerkin=galerkin)
    uzbasis = sph.Basis(s, eta, m, Lw,   Nw,   sigma=0,  alpha=1+dalpha, galerkin=galerkin)
    pbasis  = sph.Basis(s, eta, m, Lmax, Nmax, sigma=0,  alpha=2, galerkin=False)
    bases = {'up':upbasis, 'um':umbasis, 'uz':uzbasis, 'p':pbasis}

    return bases


def create_equatorial_bases(m, Lmax, Nmax, boundary_method, ns, nphi):
    s, eta = np.linspace(0,1,ns+1)[1:], np.array([0.])
    bases = create_bases(m, Lmax, Nmax, boundary_method, s, eta)

    s, phi = s[np.newaxis,:], np.linspace(0,2*np.pi,nphi+1)[:,np.newaxis]
    x, y, mode = s*np.cos(phi), s*np.sin(phi), np.exp(1j*m*phi)

    return bases, x, y, phi, mode


def expand_evectors(Lmax, Nmax, vec, bases):
    nup, num, nuz, npr = tuple(bases[f].ncoeffs for f in ['up','um','uz','p'])

    # Get the grid space vector fields
    upcoeff = vec[:nup] 
    umcoeff = vec[nup:nup+num]
    uzcoeff = vec[nup+num:nup+num+nuz]
    pcoeff = vec[nup+num+nuz:nup+num+nuz+npr]
    tau = vec[nup+num+nuz+npr:]
    print('Tau norm: {}'.format(np.linalg.norm(tau)))

    # Convert to grid space
    up = bases['up'].expand(upcoeff)
    um = bases['um'].expand(umcoeff)
    uz = bases['uz'].expand(uzcoeff)
    p  = bases['p'].expand(pcoeff)
    u, v, w = np.sqrt(0.5)*(up + um), -1j * np.sqrt(0.5)*(up - um), uz

    return u, v, w, p, tau


def plot_spectrum_callback(index, evalues, evectors, Lmax, Nmax, s, eta, bases, equatorial_bases=None):
    evalue, evector = evalues[index], evectors[:,index]
    u, v, w, p, tau = expand_evectors(Lmax, Nmax, evector, bases)

    field_indices = [0,1,2,3]
    fields = [u,v,w,p]
    field_names = ['u','v','w','p']

    fig, ax = plt.subplots(1,len(field_indices),figsize=(7,4.5))
    for i in range(len(field_indices)):
        field_index = field_indices[i]
        f = fields[field_index]
        f = f.real if np.linalg.norm(f.real) >= np.linalg.norm(f.imag) else f.imag
        sph.plotfield(s, eta, f, fig=fig, ax=ax[i], colorbar=False)
        ax[i].set_title(r'${}$'.format(field_names[field_index]))
        if i > 0:
            ax[i].set_yticklabels([])
            ax[i].set_ylabel(None)

        if field_index in [0,1,2]:
            error = max(np.linalg.norm(f[0,:]), np.linalg.norm(f[-1,:]))
            print('Boundary error, {}: {}'.format(field_names[field_index], error))

    fig.suptitle('λ = {:1.4f}'.format(evalue))
    fig.show()

    if equatorial_bases is None:
        return

    # Plot the equatorial slices
    bases, x, y, phi, mode = equatorial_bases
    u, v, w, p, tau = expand_evectors(Lmax, Nmax, evector, bases)
    fields = [u,v,w,p]

    fig, ax = plt.subplots(1,len(field_indices),figsize=(14,4))
    for i in range(len(field_indices)):
        field_index = field_indices[i]
        f = mode * fields[field_index]
        f = f.real if np.linalg.norm(f.real) >= np.linalg.norm(f.imag) else f.imag
        print(f'max({field_names[field_index]}) = {np.max(abs(f))}')

        im = ax[i].pcolormesh(x, y, f, cmap='RdBu', shading='gouraud')
        ax[i].plot(np.cos(phi), np.sin(phi), color='k', linewidth=0.5, alpha=0.5)
        ax[i].set_aspect(aspect='equal', adjustable='datalim')
        ax[i].set_title(r'${}$'.format(field_names[field_index]))

        if i > 0:
            ax[i].set_yticklabels([])
            ax[i].set_ylabel(None)

    fig.suptitle('λ = {:1.4f}'.format(evalue))
    fig.set_tight_layout(True)
    fig.show()


def plot_solution(m, Lmax, Nmax, boundary_method, Ekman, plot_fields, nev):
    # Load the data
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, directory='data', ext='.pckl', nev=nev)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    # Plot callback
    ns, neta = 256, 255
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    bases = create_bases(m, Lmax, Nmax, boundary_method, s, eta)
    equatorial_bases = create_equatorial_bases(m, Lmax, Nmax, boundary_method, ns=1024, nphi=512)
    onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, Lmax, Nmax, s, eta, bases, equatorial_bases)

    # Eigenvalue plot
    fig, ax = plot_spectrum(evalues, onpick=onpick)
    ax.set_xlim([-.1525, -0.0025])
    ax.set_ylim([-.11, .21])
    ax.set_title('Spherinder Basis')
    plot_filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, directory='figures', ext='.png')
    save_figure(plot_filename, fig)


def create_laplacian_basis(s, eta, basis):
    m, Lmax, Nmax, alpha = basis.m, basis.Lmax, basis.Nmax, basis.alpha
    return sph.Basis(s, eta, m, Lmax, Nmax, sigma=0, alpha=alpha+2, galerkin=False)


def make_laplacian_operators(basis):
    m, Lmax, Nmax, alpha = basis.m, basis.Lmax, basis.Nmax, basis.alpha

    conv = sph.convert_alpha(2, m, Lmax, Nmax, alpha, sigma=0)
    lapfull = sph.operator('laplacian')(m, Lmax, Nmax, alpha)
    laph = sao.horizontal_laplacian(m, Lmax, Nmax, alpha)
    lapz = lapfull - laph
    operators = {'Mode': conv, 'Horizontal Laplacian': laph, 'Vertical Laplacian': lapz}

    # Flush to zero
    operators = {key: sph.eliminate_zeros(op, tol=1e-13) for (key, op) in operators.items()}
    return operators


def compute_laplacian_parts(operators, lapbasis, coeffs):
    results = {key: lapbasis.expand(op @ coeffs) for (key, op) in operators.items()}

    m, s = lapbasis.m, lapbasis.s
    lapphi = -m**2/s**2 * results['Mode']
    laps = results.pop('Horizontal Laplacian') - lapphi
    results['Radial Laplacian'] = laps
    results['Azimuthal Laplacian'] = lapphi

    return results


def boundary_layer_thickness(s, eta, field, decay=0.1, relative=True):
    if eta[0] < eta[-1]:
        eta, field = [np.flipud(a) for a in [eta[:,np.newaxis], field]]
    absfield = abs(field)
    if relative:
        maxvals = np.max(absfield, axis=0)
    else:
        maxvals = np.max(absfield)
    field_scaled = absfield / maxvals  # Each column of the field scaled by its max
    indices = np.argmin(field_scaled <= decay , axis=0)
    return (1 - eta[indices].ravel()) * np.sqrt(1-s**2)


def plot_gravest_modes(m, Lmax, Nmax, boundary_method, Ekman, nev):
    # Load the data
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, directory='data', ext='.pckl', nev=nev)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    # Plot callback
#    ns, neta = 256, 257
    ns, neta = 1024, 9
    zoom = False
    if zoom:
        s, eta = np.linspace(0.4,1.0,ns+1)[1:], np.linspace(0.9,1.0,neta)
    else:
        s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    bases = create_bases(m, Lmax, Nmax, boundary_method, s, eta)
    onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, Lmax, Nmax, s, eta, bases)

    xlim, ylim, xticks, evalue_targets = None, None, None, None
    if (m,Ekman) == (14,1e-5):
        # Plot particular eigenmodes
        evalue_targets = np.array([-0.01850901249575822+0.1241580297262777j,   \
                                   -0.0373960620256699+0.2629720574084503j,    \
                                   -0.04933399647917682+0.33239767254815106j,  \
                                   -0.02587985072653081-0.036785829812208994j, \
                                   -0.03917822055255002-0.15922951557020207j,  \
                                   -0.05162747536026285-0.20299862623558654j])
        xlim, ylim = [-.2,-.0067], [-.3,.4]
        xticks = [-.2,-.15,-.1,-.05]
    elif (m,Ekman) == (30,1e-6):
        evalue_targets = np.array([-0.007054484980806183+0.060694054092827694j, \
                                   -0.014753923104231194+0.12091455276390671j,  \
                                   -0.019693538521933448+0.18512830925543186j,  \
                                   -0.010757237569379529-0.019425097197991206j, \
                                   -0.015971669036929946-0.07690418489506128j,  \
                                   -0.020517842733271896-0.10502321882336622j])
        xlim, ylim = [-.044, -.0057], [-.15,.22]
        xticks = [-.04,-.03,-.02,-.01]

    # Get the eigenvalue indices
    if evalue_targets is not None:
        evalue_indices = [np.argmin(np.abs(t-evalues)) for t in evalue_targets]
        evalue_targets = evalues[evalue_indices]

    """
    # Plot the zoomed spectrum
    fig, ax = plot_spectrum(evalues, onpick=onpick)
    if evalue_targets is not None:
        ax.plot(evalue_targets.real, evalue_targets.imag, '.', marker='s', markersize=4, color='tab:orange')

    def logstr(v):
        return r'$10^{' + str(int(np.log10(v))) + '}$'
    ax.set_title('$m$ = {}, E = {}'.format(m, logstr(Ekman)))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks)

    plot_filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, directory='figures', ext='-zoom.png')
    save_figure(plot_filename, fig)

    if evalue_targets is None:
        return
    """

    """
    nrows = 1
    ncols = 6
    fig, plot_axes = plt.subplots(nrows,ncols,figsize=(2.75*ncols,5*nrows))
    if nrows == 1:
        plot_axes = [plot_axes]
    for i, index in enumerate(evalue_indices):
        evalue, evector = evalues[index], evectors[:,index]
        u, v, w, p, tau = expand_evectors(Lmax, Nmax, evector, bases)

        relative_real = np.linalg.norm(np.real(p))/np.linalg.norm(p)
        p = p.real if relative_real > 0.5 else p.imag
        p /= np.max(abs(p))

        row, col = i//ncols, i%ncols
        ax = plot_axes[row][col]
        sph.plotfield(s, eta, p, fig=fig, ax=ax, colorbar=False)
        estr = f'{evalue:.4f}'.replace('j','i')
        ax.set_title(f'$\lambda = ${estr}')
        if col > 0:
            ax.set_yticklabels([])
            ax.set_ylabel(None)
        if row < nrows-1:
            ax.set_xticklabels([])
            ax.set_xlabel(None)

    plot_filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, directory='figures', prefix='evectors', ext='.png')
    save_figure(plot_filename, fig)
    """

    plot_laplacian = False
    plot_abs, plot_log, stretch = False, False, True
#    which_modes = slice(4, 5)
    which_modes = slice(3, 4)

    ss, ee = s[np.newaxis,:], eta[:,np.newaxis]
    zz = np.sqrt(1-ss**2)*ee
    rr = np.sqrt(ss**2 + zz**2)

    Sp, Sm, Sz = [sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=sigma, exact=True) for sigma in [+1, -1, 0]]
    dalpha = 2 if plot_laplacian else 0
    basis = [sph.Basis(s, eta, m, Lmax+2, Nmax+1, alpha=1+dalpha, sigma=sigma) for sigma in [+1, -1, 0]]
    basis += [sph.Basis(s, eta, m, Lmax, Nmax, alpha=2+dalpha, sigma=0)]

    op = sph.operator('laplacian', field='vector')(m, Lmax+2, Nmax+1, alpha=1)
    op = op + (sph.operator('laplacian')(m, Lmax, Nmax, alpha=2),)

    evalue_indices = evalue_indices[which_modes]
    for index in evalue_indices:
        evalue, evector = evalues[index], evectors[:,index]

        # Extract the coefficients from the eigenvector
        offsets = np.append(0, np.cumsum([bases[a].ncoeffs for a in ['up', 'um', 'uz', 'p']]))
        upcoeff, umcoeff, uzcoeff, pcoeff = [evector[offsets[i]:offsets[i+1]] for i in range(4)]
        if not use_full_vertical_velocity:
            uzcoeff = np.append(uzcoeff, np.zeros(len(upcoeff)-len(uzcoeff)))

        pgrid = bases['p'].expand(pcoeff)
        scale = np.max(abs(pgrid))

        # Convert from the Galerkin basis to the standard basis
        upcoeff, umcoeff, uzcoeff = [a[0] @ a[1] for a in zip([Sp, Sm, Sz], [upcoeff, umcoeff, uzcoeff])]

        # Compute the Laplacian of the fields
        coeff = [upcoeff, umcoeff, uzcoeff,pcoeff]
        if plot_laplacian:
            coeff = [op[i] @ c for (i,c) in enumerate(coeff)]
        up, um, uz, p = [basis[i].expand(coeff[i]) for i in range(4)]
        u, v, w = np.sqrt(0.5)*(up + um), -1j * np.sqrt(0.5)*(up - um), uz
        ur = (ss*u + zz*w)/rr

        titles = [r'$Δ p$', r'$e_{s} \cdot Δ u$', r'$e_{\phi} \cdot Δ u$', r'$e_{z} \cdot Δ u$', r'$e_{r} \cdot Δ u$']
        if not plot_laplacian:
            titles = [title.replace('Δ','') for title in titles]

        # Plot the boundary layer thicknesses
        if False and not plot_laplacian:
            labels = ['$e_{s} \cdot u$', r'$e_{\phi} \cdot u$', '$e_{z} \cdot u$', '$e_{r} \cdot u$']
            deltas = [boundary_layer_thickness(s, eta, field) for field in [u, v, w, ur]]
            fig, ax = plt.subplots()
            for delta, label in zip(deltas, labels):
#                ax.plot(s, delta, label=label)
                ax.semilogy(s, delta, label=label)
            ax.set_xlabel('s')
            ax.set_ylabel('z')
            ax.set_title('Boundary Layer Thickness')
            ax.grid(True)
            ax.legend(loc='upper left')
            fig.set_tight_layout(True)


        if True:
            fig, ax = plt.subplots(1,4,figsize=(15,4))
            ax[0].plot(np.arcsin(s), np.real(p[len(eta)//2,:]))
            ax[0].set_ylabel('Amplitude')
            ax[0].set_title('$Re(p)$')
            ax[1].plot(np.arcsin(s), np.real(u[len(eta)//2,:]))
            ax[1].set_title('$Re(e_{s} \cdot u)$')
            ax[2].plot(np.arcsin(s), np.real(v[len(eta)//2,:]))
            ax[2].set_title('$Re(e_{\phi} \cdot u)$')
            ax[3].plot(np.arcsin(s), np.real(ur[len(eta)//2,:]))
            ax[3].set_title('$Re(e_{r} \cdot u)$')
            for a in ax:
                a.grid(True)
                a.set_xlabel(r'$\arcsin{(s)}$')
            fig.set_tight_layout(True)

        # Plot the fields
        fig, ax = plt.subplots(1,5, figsize=(14, 5))
        for ii, field in enumerate([p, u, v, w, ur]):
            field = field/scale
            if plot_abs:
                field = abs(field)
            else:
#                norms = [np.linalg.norm(a) for a in [field.real, field.imag]]
#                plot_real = norms[0] >= norms[1]
                plot_real = True
                field = field.real if plot_real else field.imag

            if plot_log:
                field = np.log10(abs(field))

            cbar_format = 'log' if plot_laplacian and not plot_log else None
            cmap = 'RdBu'
            sph.plotfield(s, eta, field, stretch=stretch, aspect=None, fig=fig, ax=ax[ii], cmap=cmap, cbar_format=cbar_format)
            title = titles[ii]
            if plot_abs:
                title = r'$\left|' + title.replace('$','') + r'\right|$'
            else:
                re = 'Re' if plot_real else 'Im'
                title = f'${re}' + r'\left(' + title.replace('$','') + r'\right)$'
            if plot_log:
                title = r'$log_{10} {' + title.replace('$','') + r'}$'
            ax[ii].set_title(title)
            if ii > 0:
                ax[ii].set_yticklabels([])
                ax[ii].set_ylabel(None)

        suffix = 'abs' if plot_abs else 'real'
        if plot_log:
            suffix = 'log' + suffix
        if zoom:
            suffix += '-zoom'
        prefix = 'laplacian' if plot_laplacian else 'fields'
        plot_filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, directory='figures', prefix=prefix, ext=f'-{suffix}.png')
        save_figure(plot_filename, fig)


def main():
    solve = False
    plot_spy = False
    plot_evalues = True
    plot_fields = True
    boundary_method = 'galerkin'

#    m, Ekman, Lmax, Nmax, nev, evalue_target = 9, 10**-4, 24, 32, 'all', None
#    m, Ekman, Lmax, Nmax, nev, evalue_target = 14, 10**-5, 60, 60, 'all', None
#    m, Ekman, Lmax, Nmax, nev, evalue_target = 14, 10**-5, 160, 240, 1000, -0.01934+0.11290j
    m, Ekman, Lmax, Nmax, nev, evalue_target = 30, 10**-6, 80, 240, 1000, -0.0070738+0.060679j
#    m, Ekman, Lmax, Nmax, nev, evalue_target = 30, 10**-6, 160, 240, 2400, -0.0070738+0.060679j
#    m, Ekman, Lmax, Nmax, nev, evalue_target = 30, 10**-6, 80, 240, 4000, -0.0070738+0.060679j
#    m, Ekman, Lmax, Nmax, nev, evalue_target = 95, 10**-7.5, 200, 200, 1000, -0.001181+0.019639j

    print(f'Linear onset, m = {m}, Ekman = {Ekman:1.4e}')
    print(f'  Domain size: Lmax = {Lmax}, Nmax = {Nmax}')
    print(f'  Boundary method = {boundary_method}')
    print(f'  Full vertical velocity coefficients = {use_full_vertical_velocity}')

    if solve:
        solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, plot_spy, nev, evalue_target)

    if plot_fields or plot_evalues:
        plot_solution(m, Lmax, Nmax, boundary_method, Ekman, plot_fields, nev=nev)
        plt.show()


if __name__=='__main__':
    main()

