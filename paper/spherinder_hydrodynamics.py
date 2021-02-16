import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import os
import pickle

import spherinder.operators as sph
from spherinder.eigtools import eigsort, plot_spectrum, scipy_sparse_eigs
from fileio import save_data, save_figure


g_file_prefix = 'spherinder_hydrodynamics'


def coeff_sizes(Lmax, Nmax, truncate):
    lengths = [Nmax-(ell//2 if truncate else 0) for ell in range(Lmax)]
    offsets = np.append(0, np.cumsum(lengths)[:-1])
    return lengths, offsets


use_extended_pressure = True

def pressure_size(Lmax, Nmax, full=use_extended_pressure):
    return (Lmax+2, Nmax+1) if full else (Lmax, Nmax)


def matrices_galerkin(m, Lmax, Nmax, Ekman, truncate):
    """Construct matrices for X = [u(+), u(-), u(z), p]
    """
    alpha_bc, alpha_bc_s = 2, 1

    Lout, Nout = Lmax+2, Nmax+1
    Lp, Np = pressure_size(Lmax, Nmax)
    ncoeff = Lout*Nout
    ncoeff0 = Lmax*Nmax
    ncoeffp = Lp*Np
    
    # Galerkin conversion operators
    Boundp = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=+1)
    Boundm = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=-1)
    Boundz = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=0)

    # Vector laplacian
    Lapp, Lapm, Lapz = sph.operator('lap', 'vec')(m, Lout, Nout, alpha=1)
    Lapp = sph.resize(Lapp, Lout, Nout+1, Lout, Nout)
    Lapm = sph.resize(Lapm, Lout, Nout+1, Lout, Nout)
    Lapz = sph.resize(Lapz, Lout, Nout+1, Lout, Nout)

    # Vector divergence operator
    Div = sph.operator('div')(m, Lout, Nout, alpha=1)
    Div = sph.resize(Div, Lout, Nout+1, Lout, Nout)      # truncate Div . e(+)^* . u
    Divp, Divm, Divz = Div[:,:ncoeff], Div[:,ncoeff:2*ncoeff], Div[:,2*ncoeff:]

    # Pressure gradient
    Gradp, Gradm, Gradz = sph.operator('grad')(m, Lp, Np, alpha=2)
    Gradp = sph.resize(Gradp, Lp,   Np,   Lout, Nout)
    Gradm = sph.resize(Gradm, Lp,   Np+1, Lout, Nout)
    Gradz = sph.resize(Gradz, Lp-1, Np,   Lout, Nout)

    # Conversion matrices
    Cp = sph.convert_alpha(2, m, Lout, Nout, alpha=1, sigma=+1, truncate=True)
    Cm = sph.convert_alpha(2, m, Lout, Nout, alpha=1, sigma=-1, truncate=True)
    Cz = sph.convert_alpha(2, m, Lout, Nout, alpha=1, sigma=0, truncate=True)
    
    # Time derivative matrices
    M00 = Cp @ Boundp
    M11 = Cm @ Boundm
    M22 = Cz @ Boundz
    M33 = sparse.lil_matrix((ncoeff,ncoeffp))

    # i*u+ equation - spin+ velocity component
    L00 = (-2j * Cp + Ekman * Lapp) @ Boundp
    L01 = sparse.lil_matrix((ncoeff,ncoeff0))
    L02 = sparse.lil_matrix((ncoeff,ncoeff0))
    L03 = -Gradp

    # i*u- equation - spin- velocity component
    L10 = sparse.lil_matrix((ncoeff,ncoeff0))
    L11 = (2j * Cm + Ekman * Lapm) @ Boundm
    L12 = sparse.lil_matrix((ncoeff,ncoeff0))
    L13 = -Gradm

    # i*w equation - vertical velocity component
    L20 = sparse.lil_matrix((ncoeff,ncoeff0))
    L21 = sparse.lil_matrix((ncoeff,ncoeff0))
    L22 = Ekman * Lapz @ Boundz
    L23 = -Gradz

    # Divergence equation
    L30 = Divp @ Boundp
    L31 = Divm @ Boundm
    L32 = Divz @ Boundz
    L33 = sparse.lil_matrix((ncoeff,ncoeffp))

    Mmats = [M00, M11, M22, M33]
    upmats = [L00, L01, L02, L03]
    ummats = [L10, L11, L12, L13]
    uzmats = [L20, L21, L22, L23]
    pmats = [L30, L31, L32, L33]

    # Truncate the matrices
    if truncate:
        for mats in [Mmats, upmats, ummats, uzmats, pmats]:
            for i in range(3):
                mats[i] = sph.triangular_truncate(mats[i], Lmax, Nmax, Lout, Nout)
            mats[3] = sph.triangular_truncate(mats[3], Lp, Np, Lout, Nout)

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([upmats, ummats, uzmats, pmats], format=sparse_format)

    # Tau polynomials
    def tau_polynomials():
        if truncate and use_extended_pressure:
            def make_tau_column(a,b,c,d):
                return sparse.bmat([[  a,0*c,0*d],
                                    [0*a,0*c,0*d],
                                    [0*a,  c,0*d],
                                    [0*a,0*c,  d]])
        else:
            def make_tau_column(a,b,c,d):
                return sparse.bmat([[  a,0*b,0*c,0*d],
                                    [0*a,  b,0*c,0*d],
                                    [0*a,0*b,  c,0*d],
                                    [0*a,0*b,0*c,  d]])
        hstack = sparse.hstack

        Taup = sph.convert_alpha(3-alpha_bc, m, Lout, Nout, alpha=alpha_bc, sigma=+1, truncate=True)
        Taum = sph.convert_alpha(3-alpha_bc, m, Lout, Nout, alpha=alpha_bc, sigma=-1, truncate=True)
        Tauz = sph.convert_alpha(3-alpha_bc, m, Lout, Nout, alpha=alpha_bc, sigma=0, truncate=True)
        Taus = sph.convert_alpha(2-alpha_bc_s, m, Lout, Nout, alpha=alpha_bc_s, sigma=0, truncate=True)

        Ts = [Taup, Taum, Tauz, Taus]
        if truncate:
            for i in range(4):
                Ts[i] = sph.triangular_truncate(Ts[i], Lout, Nout, Lout, Nout)
        Nlengths, Noffsets = coeff_sizes(Lout, Nout, truncate)
        taup1, taum1, tauz1, taus1 = tuple(hstack([T[:,Noffsets[ell]+Nlengths[ell]-1] for ell in range(Lout-2)]) for T in Ts)
        taup2, taum2, tauz2, taus2 = tuple(T[:,Noffsets[-2]:] for T in Ts)
 
        col = make_tau_column(hstack([taup1,taup2]), hstack([taum1,taum2]), hstack([tauz1,tauz2]), hstack([taus1,taus2]))
        return col

    col = tau_polynomials()

    L = sparse.hstack([L,  col], format='csr')
    M = sparse.hstack([M,0*col], format='csr')

    return M, L


def matrices(m, Lmax, Nmax, boundary_method, Ekman, truncate):
    if boundary_method in ['galerkin']:
        return eval('matrices_' + boundary_method)(m, Lmax, Nmax, Ekman, truncate)
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


def output_filename(m, Lmax, Nmax, boundary_method, Ekman, truncate, directory, ext, prefix='evalues'):
    truncstr = '-truncated' if truncate else ''
    return make_filename_prefix(directory) + f'-{prefix}-m={m}-Lmax={Lmax}-Nmax={Nmax}-Ekman={Ekman:1.4e}-{boundary_method}{truncstr}' + ext


def solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, truncate, plot_spy, nev, evalue_target):
    # Construct the system
    print('Constructing matrix system...')
    M, L = matrices(m, Lmax, Nmax, boundary_method, Ekman, truncate)

    permute = not truncate  # FIXME: sort out permutation for truncated series
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
            'boundary_method': boundary_method, 'truncated': truncate,
            'evalues': evalues, 'evectors': evectors,
            'Ekman': Ekman}
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, truncate, directory='data', ext='.pckl')
    save_data(filename, data)


def create_bases(m, Lmax, Nmax, boundary_method, truncate, s, eta):
    if boundary_method == 'galerkin':
        galerkin = True
        dalpha = 1
        Lp, Np = pressure_size(Lmax, Nmax)
    else:
        galerkin = False
        dalpha = 0
        Lp, Np = Lmax, Nmax

    upbasis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=+1, alpha=1+dalpha, galerkin=galerkin, truncate=truncate)
    umbasis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=-1, alpha=1+dalpha, galerkin=galerkin, truncate=truncate)
    uzbasis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=0,  alpha=1+dalpha, galerkin=galerkin, truncate=truncate)
    pbasis  = sph.Basis(s, eta, m, Lp, Np, sigma=0,  alpha=2, galerkin=False, truncate=truncate)
    bases = {'up':upbasis, 'um':umbasis, 'uz':uzbasis, 'p':pbasis}

    return bases


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


def plot_spectrum_callback(index, evalues, evectors, Lmax, Nmax, s, eta, bases):
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


def plot_solution(m, Lmax, Nmax, boundary_method, Ekman, truncate, plot_fields):
    # Load the data
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, truncate, directory='data', ext='.pckl')
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    # Plot callback
    ns, neta = 256, 255
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    bases = create_bases(m, Lmax, Nmax, boundary_method, truncate, s, eta)
    onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, Lmax, Nmax, s, eta, bases)

    # Eigenvalue plot
    fig, ax = plot_spectrum(evalues, onpick=onpick)
    ax.set_title('Spherinder Basis')
    plot_filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, truncate, directory='figures', ext='.png')
    save_figure(plot_filename, fig)


def plot_gravest_modes(m, Lmax, Nmax, boundary_method, Ekman, truncate):
    # Load the data
    filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, truncate, directory='data', ext='.pckl')
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    # Plot callback
    ns, neta = 256, 255
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    bases = create_bases(m, Lmax, Nmax, boundary_method, truncate, s, eta)
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
        xlim, ylim = [-.237,-.0067], [-.9,.9]
        xticks = [-.2,-.15,-.1,-.05]
    elif (m,Ekman) == (30,1e-6):
        evalue_targets = np.array([-0.007054484980806183+0.060694054092827694j, \
                                   -0.014753923104231194+0.12091455276390671j,  \
                                   -0.019693538521933448+0.18512830925543186j,  \
                                   -0.010757237569379529-0.019425097197991206j, \
                                   -0.015971669036929946-0.07690418489506128j,  \
                                   -0.020517842733271896-0.10502321882336622j])
        xlim, ylim = [-.044, -.0057], [-.32,.32]
        xticks = [-.04,-.03,-.02,-.01]

    # Get the eigenvalue indices
    if evalue_targets is not None:
        evalue_indices = [np.argmin(np.abs(t-evalues)) for t in evalue_targets]
        evalue_targets = evalues[evalue_indices]

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

    plot_filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, truncate, directory='figures', ext='-zoom.png')
    save_figure(plot_filename, fig)

    if evalue_targets is None:
        return

    fig, ax = plt.subplots(1,len(evalue_targets),figsize=(2.75*len(evalue_targets),5))
    for i, index in enumerate(evalue_indices):
        evalue, evector = evalues[index], evectors[:,index]

        u, v, w, p, tau = expand_evectors(Lmax, Nmax, evector, bases)

        relative_real = np.linalg.norm(np.real(p))/np.linalg.norm(p)
        p = p.real if relative_real > 0.5 else p.imag
        p /= np.max(abs(p))

        sph.plotfield(s, eta, p, fig=fig, ax=ax[i], colorbar=False)
        ax[i].set_title(f'$\lambda = ${evalue_target:.4f}')
        if i > 0:
            ax[i].set_yticklabels([])
            ax[i].set_ylabel(None)

    plot_filename = output_filename(m, Lmax, Nmax, boundary_method, Ekman, truncate, directory='figures', prefix='evectors', ext='.png')
    save_figure(plot_filename, fig)


def main():
    solve = True
    plot_spy = False
    plot_evalues = True
    plot_fields = True
    boundary_method = 'galerkin'
    truncate = True

    m, Ekman, Lmax, Nmax, nev, evalue_target = 9, 10**-4, 24, 32, 'all', None
#    m, Ekman, Lmax, Nmax, nev, evalue_target = 14, 10**-5, 45, 45, 'all', None
#    m, Ekman, Lmax, Nmax, nev, evalue_target = 30, 10**-6, 60, 60, 'all', None
#    m, Ekman, Lmax, Nmax, nev, evalue_target = 30, 10**-6, 80, 240, 1000, -0.0070738+0.060679j
#    m, Ekman, Lmax, Nmax, nev, evalue_target = 95, 10**-7.5, 200, 200, 1000, -0.001181+0.019639j

    print(f'Linear onset, m = {m}, Ekman = {Ekman:1.4e}')
    print(f'  Domain size: Lmax = {Lmax}, Nmax = {Nmax}')
    print(f'  Boundary method = {boundary_method}')
    print(f'  Truncate = {truncate}, Extended pressure coefficients = {use_extended_pressure}')

    if solve:
        solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, truncate, plot_spy, nev, evalue_target)

    if plot_fields or plot_evalues:
        plot_solution(m, Lmax, Nmax, boundary_method, Ekman, truncate, plot_fields)
        plt.show()


if __name__=='__main__':
    main()

