from dedalus_sphere import ball_wrapper as ball
import dedalus.public as de
from dedalus.core.distributor import Distributor
import numpy as np
import scipy.sparse as sparse
from mpi4py import MPI
import matplotlib.pyplot as plt
import time
import boussinesq
from state_vector import StateVector
from eigtools import scipy_sparse_eigs, track_eigenpair, eigsort, plot_spectrum
from plot_tools import plotmeridionalslice, dealias, sph2cart
from interpolate import interpolate, envelope, polyfit
import scipy.signal as ss
import scipy.fft as fft
import pickle, os, glob


g_file_prefix = 'critical_rayleigh_rotating'

def build_ball(L_max, N_max):
    R_max = 3
    L_dealias = 1
    N_dealias = 1
    N_r = N_max

    # Find MPI rank
    comm = MPI.COMM_WORLD

    # Make domain
    mesh=[1]
    phi_basis = de.Fourier('phi',2*(L_max+1), interval=(0,2*np.pi),dealias=L_dealias)
    theta_basis = de.Fourier('theta', L_max+1, interval=(0,np.pi),dealias=L_dealias)
    r_basis = de.Fourier('r', N_max+1, interval=(0,1),dealias=N_dealias)
    domain = de.Domain([phi_basis,theta_basis,r_basis], grid_dtype=np.float64, mesh=mesh)

    domain.global_coeff_shape = np.array([L_max+1,L_max+1,N_max+1])
    domain.distributor = Distributor(domain,comm,mesh)

    th_m_layout  = domain.distributor.layouts[2]
    r_ell_layout = domain.distributor.layouts[1]

    m_start   = th_m_layout.slices(scales=1)[0].start
    m_end     = th_m_layout.slices(scales=1)[0].stop-1
    ell_start = r_ell_layout.slices(scales=1)[1].start
    ell_end   = r_ell_layout.slices(scales=1)[1].stop-1

    # set up ball
    N_theta = int((L_max+1)*L_dealias)
    N_r     = int((N_r+1)*N_dealias)
    B = ball.Ball(N_max,L_max,N_theta=N_theta,N_r=N_r,R_max=R_max,ell_min=ell_start,ell_max=ell_end,m_min=m_start,m_max=m_end,a=0.)

    return B, domain


def build_matrices_ell(B, Ekman, Prandtl, Rayleigh, ell_range, alpha_BC, boundary_condition):
    M, L, E = [], [], []
    for ell in ell_range:
        N = B.N_max - B.N_min(ell-B.R_max)
        M_ell, L_ell = boussinesq.matrices(B,N,ell,Ekman,Prandtl,Rayleigh,alpha_BC,implicit_buoyancy=True,implicit_base_temp=True,boundary_condition=boundary_condition)
        M.append(M_ell.astype(np.complex128))
        L.append(L_ell.astype(np.complex128))

        # Conversion matrices for Coriolis
        Em = B.op('E',N,1,ell-1).dot(B.op('E',N,0,ell-1))
        E0 = B.op('E',N,1,ell  ).dot(B.op('E',N,0,ell  ))
        Ep = B.op('E',N,1,ell+1).dot(B.op('E',N,0,ell+1))
        Z = B.op('0',N,0,ell)

        # For ell = 0 set u = 0
        if ell == 0:
            E_ell = sparse.block_diag([Z, Z, Z, Z, Z])
        else:
            E_ell = sparse.block_diag([Em, E0, Ep, Z, Z])

        # Append the tau rows and columns
        ntau = 1 if ell == 0 else 4
        nr, nc = np.shape(E_ell)
        col = np.zeros((nr, ntau))
        row = np.zeros((ntau, nc))
        E_ell = sparse.bmat([[E_ell, col], [row, np.zeros((ntau, ntau))]])

        E.append(E_ell.astype(np.complex128))

    return M, L, E


def build_unscaled_matrices(B, state_vector, m, alpha_BC, boundary_condition):
    ell_range = range(m, B.L_max+1)

    time_start = time.time()
    M, L, E = build_matrices_ell(B, Ekman=1, Prandtl=1, Rayleigh=1, ell_range=ell_range, alpha_BC=alpha_BC, boundary_condition=boundary_condition)
    time_end = time.time()
    print('    Non-rotating matrix construction took {:g} sec'.format(time_end-time_start), flush=True)

    time_start = time.time()
    Cm = boussinesq.create_coriolis_matrix(B, state_vector, m_min=m, m_max=m)
    Em = sparse.block_diag([E_ell for E_ell in E])
    Cm = Em.dot(Cm)
    time_end = time.time()
    print('    Coriolis matrix construction took {:g} sec'.format(time_end-time_start), flush=True)

    return L, M, Cm


def build_matrices(B, m, L, M, C, Ekman, Prandtl, Rayleigh):
    ell_range = range(m, B.L_max+1)
    Lscaled, Mscaled = [], []
    for ell in ell_range:
        ell_local = ell-m
        Lell, Mell = boussinesq.rescale_system(ell, L[ell_local], M[ell_local], Ekman, Prandtl, Rayleigh, inplace=False)
        Lscaled.append(Lell)
        Mscaled.append(Mell)

    Mm = sparse.block_diag([M_ell for M_ell in Mscaled])
    Lm = sparse.block_diag([L_ell for L_ell in Lscaled])

    Amat = Lm + C
    Bmat = -Mm

    return Amat, Bmat


def plot_spectral_decay(u, p, T, B, m, Ekman, Rayleigh, save_plots):
    basepath = os.path.dirname(__file__)
    filename = lambda field: os.path.join(basepath, 'figures/boussinesq-coeffs-m={}-Lmax={}-Nmax={}-Ekman={:1.4e}-Rayleigh={:1.9e}-kind={}'\
            .format(m, B.L_max, B.N_max, Ekman, Rayleigh,  field))


    if save_plots:
        def savefig(fn): plt.savefig(fn + '.png')
    else:
        def savefig(_): pass

    plt.figure()
    Tradial = []
    for ell in range(m, B.L_max + 1):
        Tradial.append(np.array([np.abs(T['c'][ell][n, m]) for n in range(np.shape(T['c'][ell])[0])]))
    factor = 1 / np.max(Tradial[0])
    for i in range(len(Tradial)):
        Tradial[i] *= factor
    for i in range(len(Tradial)):
        plt.semilogy(Tradial[i])
    plt.grid(True)
    plt.title('Magnitude of radial coefficients for each harmonic degree')
    plt.xlabel('Radial degree')
    savefig(filename('radial'))

    def plot_radial_norm(field, rank=0, a=0):
        Tell = []
        ell_range = range(m, B.L_max + 1)

        for ell in ell_range:
            nsize = np.shape(field['c'][ell])[0]
            n1 = a * (nsize // 3 ** rank)
            n2 = (a + 1) * (nsize // 3 ** rank)
            Tell.append(np.linalg.norm(field['c'][ell][n1:n2, m]))
        Tell = np.array(Tell)
        Tell /= max(Tell)
        plt.semilogy(ell_range, Tell)
        plt.grid(True)
        plt.title('Norm of radial coefficients for each harmonic degree')
        plt.xlabel('Harmonic degree l')

    plt.figure()
    plot_radial_norm(T)
    savefig(filename('ell'))


def plot_chebyshev_decay(field, r, theta, phi, m, Ekman, Rayleigh, save_plots):
    if save_plots:
        def savefig(fn): plt.savefig(fn + '.png')
    else:
        def savefig(_): pass

    def chebgrid(n):
        return np.cos((np.arange(0, n + 1) + 0.5) * np.pi / (n + 1))

    basepath = os.path.dirname(__file__)
    filename = lambda suff: os.path.join(basepath, 'figures/boussinesq-chebyshev-coeffs-m={}-Ekman={:1.4e}-Rayleigh={:1.9e}-{}'\
            .format(m, Ekman, Rayleigh, suff))

    sfix, zfix = 0.5768, 0.0
    method = 'cubic'
    nx, nz = 256, 256

    # Create evaluation points
    z_eval = chebgrid(nz)
    z_eval *= (1 - sfix**2)**.5 * .998
    xi = np.array([[sfix, z] for z in z_eval])

    # Interpolate
    zresult = interpolate(field, r, theta, phi, xi, method=method)
    dctz = fft.dct(zresult)

    # Create evaluation points
    x_eval = 0.5*(chebgrid(nx)+1)
    x_eval *= (1 - zfix**2)**.5 * .998
    xi = np.array([[x, zfix] for x in x_eval])

    # Interpolate
    xresult = interpolate(field, r, theta, phi, xi, method=method)
    dctx = fft.dct(xresult)

    fig, plot_axes = plt.subplots(1, 2, figsize=[12.8,4.8])
    plot_axes[0].plot(z_eval, zresult)
    plot_axes[0].grid()
    plot_axes[0].set_xlabel('z')
    plot_axes[0].set_ylabel('T')
    plot_axes[0].set_title('Axial Temperature Field, s={}'.format(sfix))
    plot_axes[1].semilogy(abs(dctz)/max(abs(dctz)))
    plot_axes[1].grid()
    plot_axes[1].set_xlabel('n')
    plot_axes[1].set_ylabel('log magnitude')
    plot_axes[1].set_title('Chebyshev Coefficient Magnitude')
    savefig(filename('axial'))

    fig, plot_axes = plt.subplots(1, 2, figsize=[12.8,4.8])
    plot_axes[0].plot(x_eval, xresult)
    plot_axes[0].grid()
    plot_axes[0].set_xlabel('s')
    plot_axes[0].set_ylabel('T')
    plot_axes[0].set_title('Radial Temperature Field, z={}'.format(zfix))
    plot_axes[1].semilogy(abs(dctx)/max(abs(dctx)))
    plot_axes[1].grid()
    plot_axes[1].set_xlabel('n')
    plot_axes[1].set_ylabel('log magnitude')
    plot_axes[1].set_title('Chebyshev Coefficient Magnitude')
    savefig(filename('radial'))


def plot_fields(u, p, T, B, m, domain, Ekman, Rayleigh, save_plots, plot_resolution, plot_dpi):
    if save_plots:
        def savefig(fn): plt.savefig(fn + '.png', dpi=plot_dpi)
    else:
        def savefig(_): pass

    # Dealias for plotting
    L_factor, N_factor = max(plot_resolution // (B.L_max + 1), 1), max(plot_resolution // (B.N_max + 1), 1)

    basepath = os.path.dirname(__file__)
    filename = lambda field, sl: os.path.join(basepath,'figures/stretch/boussinesq-m={}-Lmax={}-Nmax={}-field={}-slice={}-Ekman={:1.6e}-Rayleigh={:1.6e}'.format(
        m, B.L_max, B.N_max, field, sl, Ekman, Rayleigh))


    # Plot settings
    angle = 0.
    stretch = False

    """
    # Compute the vorticity
    om = ball.TensorField_3D(1, B, domain)
    om.layout = 'c'
    for ell in range(m, B.L_max + 1):
        B.curl(ell, 1, u['c'][ell], om['c'][ell])
    """

    # Dealias the fields
    T, r, theta, phi = dealias(B, domain, T, L_factor=L_factor, N_factor=N_factor)
    T = T['g'][0]
    """
    u, r, theta, phi = dealias(B, domain, u, L_factor=L_factor, N_factor=N_factor)
    ur, utheta, uphi = u['g'][0], u['g'][1], u['g'][2]

    # Cartesian velocities
    _, _, uz = sph2cart(u, theta, phi)

    # Dealias the vorticity field
    om, _, _, _ = dealias(B, domain, om, L_factor=L_factor, N_factor=N_factor)
    omz = om['g'][0] * np.cos(theta) - om['g'][1] * np.sin(theta)

    plotmeridionalslice(uphi, r, theta, phi, angle=angle, stretch=stretch)
    plt.title('Meridional Slice, $u_ϕ$')
    savefig(filename('uphi', 'm'))

    plotmeridionalslice(uz, r, theta, phi, angle=angle, stretch=stretch)
    plt.title('Meridional Slice, $u_z$')
    savefig(filename('uz', 'm'))

    plotmeridionalslice(omz, r, theta, phi, angle=angle, stretch=stretch)
    plt.title('Meridional Slice, $ω_z$')
    savefig(filename('omz', 'm'))
    """

    plotmeridionalslice(T, r, theta, phi, angle=angle, stretch=stretch)
    plt.title('Meridional Slice, $T$')
    savefig(filename('T', 'm'))

    """
    # Normalized kinetic energy
    ke = np.log10(0.5 * (u['g'][0] ** 2 + u['g'][1] ** 2 + u['g'][2] ** 2))
    ke -= np.max(ke)
    truncate_level = -8
    ketrunc = np.where(ke < truncate_level, np.nan, ke)

    plotmeridionalslice(ketrunc, r, theta, phi, angle=angle, stretch=stretch, cmap='RdBu_r')
    plt.title('Meridional Slice, $log_{10}$(Kinetic Energy), Truncated')
    savefig(filename('ke', 'm'))
    """


def rayleigh_bisection(B, m, Lunscaled, Munscaled, Cor, lam, v, Ekman, Prandtl, Rayleigh, bracket_scale=1.1,
                       newton_tol=1e-9, bisect_tol=1e-3, max_newton_iters=12, max_bisect_iters=12, verbose=False):
    if max_bisect_iters == 0:
        return lam, v, Rayleigh

    matsolver = 'SuperluNaturalSpsolve'

    eig = lambda Amat, Bmat, lam, v: track_eigenpair(Amat, Bmat, lam, v, matsolver=matsolver, tol=newton_tol,
                                                     maxiter=max_newton_iters, verbose=verbose)

    # Check current Rayleigh number
    print('  Building matrices to bracket the critical Rayleigh number...', flush=True)
    Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)

    print('  Computing initial eigenpair...', flush=True)
    lam, v = eig(Amat, Bmat, lam, v)

    print('  Bracketing critical Rayleigh number for Ekman = {:1.5e},  Eigenvalue = {:1.5e}...'.format(Ekman, lam), flush=True)
    Ra_min, Ra_max = Rayleigh, Rayleigh
    moveup = lam.real < 0
    if moveup:
        while lam.real <= 0:
            Rayleigh *= bracket_scale
            Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)

            lam, v = eig(Amat, Bmat, lam, v)
            Ra_min = Ra_max
            Ra_max = Rayleigh
            print('    Rayleigh = {:1.5e},  Eigenvalue = {:1.5e}...'.format(Rayleigh, lam), flush=True)
    else:
        while lam.real >= 0:
            Rayleigh /= bracket_scale
            Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)

            lam, v = eig(Amat, Bmat, lam, v)
            Ra_max = Ra_min
            Ra_min = Rayleigh
            print('    Rayleigh = {:1.5e},  Eigenvalue = {:1.5e}...'.format(Rayleigh, lam), flush=True)

    print('  Bisection brackets for critical Rayleigh number: {}'.format((Ra_min,Ra_max)), flush=True)
    iter, done = 0, False
    while not done:
        iter += 1
        Rayleigh = 0.5 * (Ra_min + Ra_max)

        # Build the matrices for the updated Rayleigh number
        print('  Bisection Iteration: {:3d},  Building matrices for Rayleigh = {:1.9e}'.format(iter, Rayleigh), flush=True)
        Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)

        # Track the eigenpair as Rayleigh changed
        print('    Tracking eigenpair...', flush=True)
        lam, v = eig(Amat, Bmat, lam, v)

        # Update the bisection brackets
        if np.real(lam) > 0:
            Ra_max = Rayleigh
        else:
            Ra_min = Rayleigh

        print('    Ekman: {:1.5e},  Rayleigh number: {:1.9e},  λ = {: 1.9e}'.format(iter, Ekman, Rayleigh, lam), flush=True)
        done = np.abs(np.real(lam)) <= bisect_tol or iter >= max_bisect_iters

    v = np.reshape(np.asarray(v.ravel()), np.prod(np.shape(v)))
    return lam, v, Rayleigh


def checkdir(filename):
    path = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(path):
        os.makedirs(path)


def save_data(filename, data):
    checkdir(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def save_figure(filename, fig):
    checkdir(filename)
    fig.savefig(filename)


def make_filename_prefix(directory='data'):
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), directory))
    abspath = os.path.join(basepath, g_file_prefix)
    return os.path.join(abspath, g_file_prefix)


def output_filename(m, Lmax, Nmax, boundary_condition, Ekman, Prandtl, Rayleigh, directory, ext, prefix='evalues'):
    return make_filename_prefix(directory) + f'-{prefix}-m={m}-Lmax={Lmax}-Nmax={Nmax}-Ekman={Ekman:1.4e}-Prandtl={Prandtl:1.4e}-Rayleigh={Rayleigh:1.4e}-{boundary_condition}' + ext


def solve_eigenproblem(B, m, domain, config, nev, boundary_condition='stress-free',
                              thermal_forcing_factor=1.0):
    print('Computing Critical Rayleigh Number', flush=True)
    print('  Boussinesq ball dimensions: m = {}, L_max = {}, N_max = {}'.format(m, B.L_max, B.N_max), flush=True)

    plot_evec = True
    plot_coeff_decay = False
    save_plots = False
    save_evec = False
    plot_resolution = 256
    plot_dpi = 600

    # Get reduced nondimensional parameters from config
    Ekman, Rayleigh, omega = config['Ekman'], config['Rayleigh'], config['omega']

    # Rescale parameters
    omega /= Ekman**(2/3)
    Prandtl = 1
    Rayleigh = thermal_forcing_factor * Rayleigh / Ekman**(1/3)
    alpha_BC = 2

    lamtarget = 1j*omega

    print('  Constructing the full system, m = {}, Ekman = {}, Rayleigh = {}...'.format(m, Ekman, Rayleigh), flush=True)
    ntau = lambda ell: 1 if ell == 0 else 4
    fields = [('u', 1), ('p', 0), ('T', 0)]
    state_vector = StateVector(B, 'mlr', fields, ntau=ntau, m_min=m, m_max=m)

    time_start = time.time()
    Lunscaled, Munscaled, Cor = build_unscaled_matrices(B, state_vector, m, alpha_BC, boundary_condition)

    t1 = time.time()
    Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)
    t2 = time.time()
    print('    Non-dimensional scaling took {:g} sec'.format(t2-t1), flush=True)

    time_end = time.time()
    print('    Complete system construction took {:g} sec'.format(time_end-time_start), flush=True)

    print('  Solving sparse eigenproblem for m = {}, L_max = {}, N_max = {}, size {}x{}'.format(m, B.L_max, B.N_max, np.shape(Amat)[0], np.shape(Amat)[1]), flush=True)
    if nev == 'all':
        evalues, evectors = eigsort(Amat.todense(), Bmat.todense(), profile=True)
    else:
        evalues, evectors = scipy_sparse_eigs(Amat, Bmat, N=nev, target=lamtarget, profile=True)

    # Output data
    data = {'m': m, 'Lmax': B.L_max, 'Nmax': B.N_max,
            'boundary_condition': boundary_condition,
            'evalues': evalues, 'evectors': evectors,
            'Ekman': Ekman, 'Prandtl': Prandtl, 'Rayleigh': Rayleigh}
    filename = output_filename(m, B.L_max, B.N_max, boundary_condition, Ekman, Prandtl, Rayleigh, directory='data', ext='.pckl')
    save_data(filename, data)


def plot_spectrum_callback(index, evalues, evectors, B, m, domain):
    plot_velocity = True
    evalue, evector = evalues[index], evectors[:,index]

    ntau = lambda ell: 1 if ell == 0 else 4
    u = ball.TensorField_3D(1, B, domain)
    p = ball.TensorField_3D(0, B, domain)
    T = ball.TensorField_3D(0, B, domain)
    state_vector = StateVector(B, 'mlr', [('u',1),('p',0),('T',0)], ntau=ntau, m_min=m, m_max=m)
    state_vector.unpack(evector, [u, p, T])

    # Upsample the result
    res = 256
    L_factor, N_factor = res // (B.L_max + 1), res // (B.N_max + 1)

    p, r, theta, phi = dealias(B, domain, p, L_factor=L_factor, N_factor=N_factor)
    T, _, _, _       = dealias(B, domain, T, L_factor=L_factor, N_factor=N_factor)
    names, fields = ['p', 'T'], [p['g'][0], T['g'][0]]
    if plot_velocity:
        u, _, _, _ = dealias(B, domain, u, L_factor=L_factor, N_factor=N_factor)
        names += ['u_r', 'u_θ', 'u_φ']
        fields += [u['g'][i] for i in [0,1,2]]

    angle = 0.
    # Plot velocity
    for index, name in enumerate(names):
        fig, ax = plotmeridionalslice(fields[index], r, theta, phi, angle=angle)
        fig.suptitle('${}$,  λ = {:1.4e}'.format(name, evalue))
        fig.show()


def plot_solution(B, m, domain, config, boundary_condition, thermal_forcing_factor):
    # Get reduced nondimensional parameters from config
    Ekman, Rayleigh, omega = config['Ekman'], config['Rayleigh'], config['omega']

    # Rescale parameters
    omega /= Ekman**(2/3)
    Prandtl = 1
    Rayleigh = thermal_forcing_factor * Rayleigh / Ekman**(1/3)

    # Load the data
    filename = output_filename(m, B.L_max, B.N_max, boundary_condition, Ekman, Prandtl, Rayleigh, directory='data', ext='.pckl')
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    plot_fields = True
    if plot_fields:
        onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, B, m, domain)
    else:
        onpick = None

    fig, ax = plot_spectrum(evalues, onpick)
    ax.set_title('Hydrodynamics Eigenvalues')
    ax.set_xlabel('Real(λ)')
    ax.set_ylabel('Imag(λ)')

    plot_filename = output_filename(m, B.L_max, B.N_max, boundary_condition, Ekman, Prandtl, Rayleigh, directory='figures', ext='.png')
    save_figure(plot_filename, fig)

    fig.show()


def compute_eigensolutions():
    import warnings
    warnings.simplefilter("ignore")

    solve = True
    plot = True

    configs = [{'Ekman': 10**-4,   'm': 6,  'omega': -.43346, 'Rayleigh': 5.1549, 'Lmax': 32,  'Nmax': 31},
               {'Ekman': 10**-4.5, 'm': 9,  'omega': -.44276, 'Rayleigh': 4.7613, 'Lmax': 32,  'Nmax': 31},
               {'Ekman': 10**-5,   'm': 14, 'omega': -.45715, 'Rayleigh': 4.5351, 'Lmax': 48,  'Nmax': 47},
               {'Ekman': 10**-5.5, 'm': 20, 'omega': -.45760, 'Rayleigh': 4.3937, 'Lmax': 92,  'Nmax': 63},
               {'Ekman': 10**-6,   'm': 30, 'omega': -.46394, 'Rayleigh': 4.3021, 'Lmax': 122, 'Nmax': 83},
               {'Ekman': 10**-6.5, 'm': 44, 'omega': -.46574, 'Rayleigh': 4.2416, 'Lmax': 152, 'Nmax': 101},
               {'Ekman': 10**-7,   'm': 65, 'omega': -.46803, 'Rayleigh': 4.2012, 'Lmax': 232, 'Nmax': 171},
               {'Ekman': 10**-7.5, 'm': 95, 'omega': -.46828, 'Rayleigh': 4.1742, 'Lmax': 296, 'Nmax': 201}]

    boundary_condition = 'no-slip'
    # boundary_condition = 'stress-free'
    nev = 3
    # nev = 'all'
    thermal_forcing_factor = 1

    configs = [configs[4]]
    for config in configs:
        # Build the domain
        m, L_max, N_max = config['m'], config['Lmax'], config['Nmax']
        B, domain = build_ball(L_max=L_max, N_max=N_max)

        # Solve the eigenproblem
        if solve:
            solve_eigenproblem(B, m, domain, config, nev=nev, boundary_condition=boundary_condition,
                               thermal_forcing_factor=thermal_forcing_factor)

        # Plot the solution
        if plot:
            plot_solution(B, m, domain, config, boundary_condition, thermal_forcing_factor)

    plt.show()


def extract_radial_slice(field):
    Ntheta = np.shape(field)[1]
    signal = field[0,:,:]
    if Ntheta % 2 == 1:
        signal = 0.5*(signal[Ntheta//2-1,:] + signal[Ntheta//2,:])
    else:
        signal = signal[Ntheta//2,:]
    return signal


def count_zero_crossings(a):
    return np.where(np.diff(np.sign([i for i in a if i])))[0].shape[0]


def analyze_radial_field(r, theta, phi, field, Ekman, fieldname, doplot=True, npeaks=5, peak_level=10**-8, plotonly=False):
    fit_kind = 'envelope'        # one of ['lsq', 'peak', 'envelope']
    envelope_kind = 'peak'      # one of ['peak', 'analytic']

    r, theta, phi = r.flatten(), theta.flatten(), phi.flatten()
    Nr = len(r)

    # Extract the phi=0, theta=pi/2 radial slice
    signal = extract_radial_slice(field)

    # Normalize so the peak points up
    if np.max(signal) < -np.min(signal):
        signal *= -1
    signal /= np.max(abs(signal))
    fitsig = np.log10(abs(signal))

    # Compute the width of the peak
    index_peak_start = np.argmax(abs(signal) > peak_level)
    index_peak_end = Nr - 1 - np.argmax(abs(signal[::-1]) > peak_level)
    peak_start, peak_end = r[index_peak_start], r[index_peak_end]
    peak_width = peak_end - peak_start
    peak_midpoint = (peak_start + peak_end)/2

    # Plot
    ekstr = '$10^{{{}}}$'.format(np.around(np.log10(Ekman), 1))
    if plotonly:
        fig, ax = plt.subplots()
        ax.plot(r, signal)
        ax.set_xlabel('s')
        ax.set_ylabel(fieldname)
        ax.set_title('{},  Ekman = {}'.format(fieldname, ekstr))
        ax.grid()
        fig.set_tight_layout(True)

        outputdir = os.path.join(os.path.dirname(__file__), 'figures')
        filename = os.path.join(outputdir, 'boussinesq_radial_slice-Ekman={:1.3e}-field={}.png'.format(Ekman, fieldname))
        plt.savefig(filename, dpi=200)
        return

    # Compute the polynomial or envelope fit
    if fit_kind == 'peak' or fit_kind == 'lsq':
        p, fit = polyfit(r, fitsig, fit_kind, npeaks=npeaks, first=index_peak_start, final=index_peak_end)
        fit = 10**np.polyval(p, r)
        peak_location = -p[1]/(2*p[0])
        peak_height = 10**np.polyval(p, peak_location)

    elif fit_kind == 'envelope':
        env = envelope(r, fitsig, envelope_kind, first=index_peak_start, final=index_peak_end, usehull=False)
        fit = 10**env
        peak_index = index_peak_start+np.argmax(fit[index_peak_start:index_peak_end+1])
        peak_location, peak_height = r[peak_index], fit[peak_index]

    else:
        raise ValueError('Unknown fit_kind')

    peak_indices, _ = ss.find_peaks(signal[index_peak_start:index_peak_end+1])
    npeaks = len(peak_indices)
    print('Ekman = {:1.3e},  Field = {},  Peak Location: s = {:1.6f},  Peak Width: s = {:1.6e},  Peak Midpoint: {:1.4f},'
          '  Num Peaks: {}'
          .format(Ekman, fieldname, peak_location, peak_width, peak_midpoint, npeaks))

    if doplot:
        fig, plot_axes = plt.subplots()
        plot_axes.semilogy(r, abs(signal))
        plot_axes.plot(r, fit, '--', color='C2')
        plot_axes.plot(peak_location, peak_height, 'x', color='k')
        plot_axes.grid()

        plot_axes.set_title('log({}),  Ekman = {}'.format(fieldname, ekstr))
        plot_axes.set_xlabel('s')
        plot_axes.set_ylabel('log({})'.format(fieldname))

        eps = 0.042
        xlim = [-eps, 1+eps]
        plot_axes.plot(xlim, [peak_level, peak_level], '--k')
        plot_axes.set_xlim(xlim)
        plot_axes.set_ylim([10**-9, 10])
        fig.set_tight_layout(True)

        outputdir = os.path.join(os.path.dirname(__file__), 'figures')
        filename = os.path.join(outputdir, 'boussinesq_radial_slice-Ekman={:1.3e}-field={}.png'.format(Ekman, fieldname))
        plt.savefig(filename, dpi=200)

    result = {'location': peak_location, 'width': peak_width, 'midpoint': peak_midpoint,
              'start': peak_start, 'end': peak_end, 'npeaks': npeaks}
    return result


def analyze_eigensolutions():
    resolution = 256
    peak_level = 10 ** -3

    datadir = os.path.join(os.path.dirname(__file__), 'data')
    datafiles = glob.glob(datadir+"/*.pckl")
    print(datafiles)

    # Load the pickle data
    alldata = np.array([pickle.load(open(filename, 'rb')) for filename in datafiles])
    Ekmans = [data['config']['Ekman'] for data in alldata]
    indices = np.argsort(Ekmans)[::-1]
    alldata = alldata[indices]

    # Process each data
    Ekmans, peak_locations, peak_widths, peak_midpoints, peak_counts = [], [], [], [], []
    for data in alldata:
        config = data['config']
        m, L_max, N_max, Ekman, Rayleigh = config['m'], config['Lmax'], config['Nmax'], config['Ekman'], config['Rayleigh']
        Ekmans.append(Ekman)

        # Construct the domain
        B, domain = build_ball(L_max=L_max, N_max=N_max)
        ntau = lambda ell: 1 if ell == 0 else 4
        fields = [('u', 1), ('p', 0), ('T', 0)]
        state_vector = StateVector(B, 'mlr', fields, ntau=ntau, m_min=m, m_max=m)

        # Unpack the solution
        evec = data['evector']
        u = ball.TensorField_3D(1, B, domain)
        p = ball.TensorField_3D(0, B, domain)
        T = ball.TensorField_3D(0, B, domain)
        state_vector.unpack(evec, [u, p, T])

        # Plot the temperature field
        L_factor, N_factor = 1, np.ceil(resolution / (B.N_max + 1))
        T, r, theta, phi = dealias(B, domain, T, L_factor=L_factor, N_factor=N_factor)
        analyze_radial_field(r, theta, phi, T['g'][0], Ekman, fieldname='T', peak_level=peak_level, plotonly=True)

        # Dealias and compute kinetic energy
        L_factor, N_factor = 1, np.ceil(resolution / (B.N_max + 1))
        u, r, theta, phi = dealias(B, domain, u, L_factor=L_factor, N_factor=N_factor)
        ke = 0.5*(u['g'][0]**2 + u['g'][1]**2 + u['g'][2]**2)

        # Find peak location and width for the signal
        npeaks = 5
        result = analyze_radial_field(r, theta, phi, ke, Ekman, fieldname='KE', doplot=True,
                                      npeaks=npeaks, peak_level=peak_level)

        peak_locations.append(result['location'])
        peak_widths.append(result['width'])
        peak_midpoints.append(result['midpoint'])
        peak_counts.append(result['npeaks'])

    ratios = np.zeros(len(alldata))
    for i in range(len(alldata)):
        m = alldata[i]['config']['m']
        mscale = 2*np.pi/m * peak_locations[i]
        sscale = peak_widths[i]/(peak_counts[i]/2)
        ratios[i] = mscale/sscale

    plot_azimuth_to_radius = True
    if not plot_azimuth_to_radius:
        ratios = 1/ratios

    fig, plot_axes = plt.subplots()
    plot_axes.semilogx(Ekmans, ratios, '--x')
    plot_axes.set_xlabel('Ekman')
    plot_axes.set_ylabel('Ratio')
    if plot_azimuth_to_radius:
        plot_axes.set_title('Azimuthal to Radial Scale Length Ratio')
    else:
        plot_axes.set_title('Radial to Azimuthal Scale Length Ratio')
    plot_axes.grid()
    fig.set_tight_layout(True)
    outputdir = os.path.join(os.path.dirname(__file__), 'figures')
    filename = os.path.join(outputdir, 'boussinesq_radial_vs_azimuthal_scaling.png')
    plt.savefig(filename)

    fig, plot_axes = plt.subplots(1,2, figsize=(9,4))
    plot_axes[0].semilogx(Ekmans, peak_locations, '--x', label='estimate')
    plot_axes[0].semilogx(Ekmans, peak_midpoints, '--x', label='midpoint')
    plot_axes[0].set_title('Peak Location')
    plot_axes[0].legend()

    plot_axes[1].semilogx(Ekmans, peak_widths, '--x', label='KE')
    plot_axes[1].set_title('Peak Width')
    for ax in plot_axes:
        ax.set_xlabel('Ekman')
        ax.set_ylabel('s')
        ax.grid()
    fig.set_tight_layout(True)
    outputdir = os.path.join(os.path.dirname(__file__), 'figures')
    filename = os.path.join(outputdir, 'boussinesq_radial_scaling.png')
    plt.savefig(filename)

    fig, plot_axes = plt.subplots()
    sm = 0.5915
    plot_axes.semilogx(Ekmans, np.array(peak_locations) - sm, '--x', label='KE')
    plot_axes.set_title('Distance from Peak Location to Jones Solution $s_M = {}$'.format(sm))
    plot_axes.set_xlabel('Ekman')
    plot_axes.set_ylabel('$s_c - s_M$')
    plot_axes.grid()
    fig.set_tight_layout(True)

    outputdir = os.path.join(os.path.dirname(__file__), 'figures')
    filename = os.path.join(outputdir, 'boussinesq_radial_scaling-difference.png')
    plt.savefig(filename)

    plt.show()


if __name__=='__main__':
    compute_eigensolutions()
    # analyze_eigensolutions()
