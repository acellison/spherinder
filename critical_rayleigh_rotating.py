from dedalus_sphere import ball_wrapper as ball
from dedalus_sphere import ball128
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

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_ball_lazy(m, L_max, N_max):
    R_max = 3
    m_start, m_end = m, m
    ell_start, ell_end = m, L_max

    # Don't actually construct the ball, just hold the configuraiton in an attribute dictionary.
    # Saves on memory while performing the eigensolve
    B = AttrDict({'N_max':N_max, 'L_max':L_max, 'R_max':R_max, 'ell_min':ell_start, 'ell_max':ell_end, 'm_min':m_start, 'm_max':m_end, 'a':0.})

    # Make domain
    domain = None

    return B, domain


def build_ball_full(m, L_max, N_max, L_factor=1, N_factor=1):
    R_max = 3
    L_dealias = L_factor
    N_dealias = N_factor
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
    def op(op_name,N,k,ell,a=B.a,dtype=np.float64):
        return ball128.operator(op_name,N,k,ell,a=a).astype(dtype)

    M, L, E = [], [], []
    for ell in ell_range:
        N = B.N_max - ball128.N_min(ell-B.R_max)
        M_ell, L_ell = boussinesq.matrices(B,N,ell,Ekman,Prandtl,Rayleigh,alpha_BC,implicit_buoyancy=True,implicit_base_temp=True,boundary_condition=boundary_condition)
        M.append(M_ell.astype(np.complex128))
        L.append(L_ell.astype(np.complex128))

        # Conversion matrices for Coriolis
        Em = op('E',N,1,ell-1).dot(op('E',N,0,ell-1))
        E0 = op('E',N,1,ell  ).dot(op('E',N,0,ell  ))
        Ep = op('E',N,1,ell+1).dot(op('E',N,0,ell+1))
        Z = op('0',N,0,ell)

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
    print('Solving Linear Onset Eigenproblem', flush=True)
    print('  Boussinesq ball dimensions: m = {}, L_max = {}, N_max = {}'.format(m, B.L_max, B.N_max), flush=True)

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

    if np.any(~np.isfinite(Amat.tocsr().data)):
        warning('Warning: non-finite values in A matrix')
    if np.any(~np.isfinite(Bmat.tocsr().data)):
        warning('Warning: non-finite values in B matrix')

    time_end = time.time()
    print('    Complete system construction took {:g} sec'.format(time_end-time_start), flush=True)

    print('  Solving sparse eigenproblem for m = {}, L_max = {}, N_max = {}, size {}x{}'.format(m, B.L_max, B.N_max, np.shape(Amat)[0], np.shape(Amat)[1]), flush=True)
    if nev == 'all':
        evalues, evectors = eigsort(Amat.todense(), Bmat.todense(), profile=True)
    else:
#        matsolver = 'UmfpackFactorized'
        matsolver = 'SuperluColamdFactorized'
#        matsolver = None
        evalues, evectors = scipy_sparse_eigs(Amat, Bmat, N=nev, target=lamtarget, matsolver=matsolver, profile=True)

    # Output data
    data = {'m': m, 'Lmax': B.L_max, 'Nmax': B.N_max,
            'boundary_condition': boundary_condition,
            'evalues': evalues, 'evectors': evectors,
            'Ekman': Ekman, 'Prandtl': Prandtl, 'Rayleigh': Rayleigh}
    filename = output_filename(m, B.L_max, B.N_max, boundary_condition, Ekman, Prandtl, Rayleigh, directory='data', ext='.pckl')
    save_data(filename, data)


def plot_spectrum_callback(index, evalues, evectors, B, m, domain):
    plot_pressure = True
    plot_temperature = False
    plot_velocity = False
    evalue, evector = evalues[index], evectors[:,index]

    ntau = lambda ell: 1 if ell == 0 else 4
    state_vector = StateVector(B, 'mlr', [('u',1),('p',0),('T',0)], ntau=ntau, m_min=m, m_max=m)

    names, fields = [], []

    # Collect fields, converting to grid space
    if plot_pressure:
        p = ball.TensorField_3D(0, B, domain)
        state_vector.unpack(evector, {'p':p})
        names += ['p']
        fields += [p['g'][0]]
    if plot_temperature:
        T = ball.TensorField_3D(0, B, domain)
        state_vector.unpack(evector, {'T':T})
        names += ['T']
        fields += [T['g'][0]]
    if plot_velocity:
        u = ball.TensorField_3D(1, B, domain)
        state_vector.unpack(evector, {'u':u})
        names += ['u_r', 'u_θ', 'u_φ']
        fields += [u['g'][i] for i in [0,1,2]]

    # Get coordinates
    r, theta, phi = B.grid(1), B.grid(0), domain.grid(0, scales=domain.dealias)
    r = r.reshape((1,1,np.prod(np.shape(r))))
    theta = theta.reshape((1,np.prod(np.shape(theta)),1))

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
        # Build the full domain for field plotting
        print(f'Building ball: m = {m}, Lmax = {B.L_max}, Nmax = {B.N_max}', flush=True)
        res = max(256, max(B.L_max+1, B.N_max+1))
        L_factor, N_factor = res // (B.L_max + 1), res // (B.N_max + 1)

        B, domain = build_ball_full(m, B.L_max, B.N_max, L_factor, N_factor)
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
#    import warnings
#    warnings.simplefilter("ignore")

    solve = False
    plot = True

    configs = [{'Ekman': 10**-4,   'm': 6,  'omega': -.43346, 'Rayleigh': 5.1549, 'Lmax': 32,  'Nmax': 31},
               {'Ekman': 10**-4.5, 'm': 9,  'omega': -.44276, 'Rayleigh': 4.7613, 'Lmax': 32,  'Nmax': 31},
               {'Ekman': 10**-5,   'm': 14, 'omega': -.45715, 'Rayleigh': 4.5351, 'Lmax': 48,  'Nmax': 47},
               {'Ekman': 10**-5.5, 'm': 20, 'omega': -.45760, 'Rayleigh': 4.3937, 'Lmax': 92,  'Nmax': 63},
               {'Ekman': 10**-6,   'm': 30, 'omega': -.46394, 'Rayleigh': 4.3021, 'Lmax': 122, 'Nmax': 83},
               {'Ekman': 10**-6.5, 'm': 44, 'omega': -.46574, 'Rayleigh': 4.2416, 'Lmax': 152, 'Nmax': 101},
               {'Ekman': 10**-7,   'm': 65, 'omega': -.46803, 'Rayleigh': 4.2012, 'Lmax': 232, 'Nmax': 171},
               {'Ekman': 10**-7.5, 'm': 95, 'omega': -.46828, 'Rayleigh': 4.1742, 'Lmax': 296, 'Nmax': 201},
               {'Ekman': 10**-8,   'm': 139,'omega': -.43507, 'Rayleigh': 4.1527, 'Lmax': 500, 'Nmax': 500},
               {'Ekman': 10**-10,  'm': 646,'omega': -.43507, 'Rayleigh': 4.1527, 'Lmax': 1200, 'Nmax': 600}
               ]

    boundary_condition = 'no-slip'
    # boundary_condition = 'stress-free'
    nev = 10
    # nev = 'all'
    thermal_forcing_factor = 1

    configs = [configs[1]]
    for config in configs:
        # Extract the domain parameters
        m, L_max, N_max = config['m'], config['Lmax'], config['Nmax']
        if L_max < m:
            raise ValueError('No angular resolution: L_max (={}) is too small'.format(L_max))

        B, domain = build_ball_lazy(m, L_max, N_max)

        # Solve the eigenproblem
        if solve:
            solve_eigenproblem(B, m, domain, config, nev=nev, boundary_condition=boundary_condition,
                               thermal_forcing_factor=thermal_forcing_factor)

        # Plot the solution
        if plot:
            plot_solution(B, m, domain, config, boundary_condition, thermal_forcing_factor)

    plt.show()


if __name__=='__main__':
    compute_eigensolutions()
