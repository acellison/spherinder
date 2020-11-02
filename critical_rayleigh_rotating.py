import pickle, os, glob
import copy, time
import multiprocessing as mp

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from dedalus_sphere import ball128, sphere128
import boussinesq
from state_vector import StateVector
from eigtools import scipy_sparse_eigs, eigsort, plot_spectrum


g_file_prefix = 'critical_rayleigh_rotating'


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_ball(m, L_max, N_max):
    """Don't actually construct the ball, just hold the configuration in an attribute dictionary.
       Saves memory while performing the eigensolve"""
    R_max = 3
    m_start, m_end = m, m
    ell_start, ell_end = m, L_max

    B = AttrDict({'N_max':N_max, 'L_max':L_max, 'R_max':R_max, 'ell_min':ell_start, 'ell_max':ell_end, 'm_min':m_start, 'm_max':m_end, 'a':0.})

    # Make domain
    domain = None

    return B, domain


def build_matrices_ell_fun(ell, B, Ekman, Prandtl, Rayleigh, alpha_BC, boundary_condition):
    def op(op_name,N,k,ell,a=B.a,dtype=np.float64):
        return ball128.operator(op_name,N,k,ell,a=a).astype(dtype)

    N = B.N_max - ball128.N_min(ell-B.R_max)
    M_ell, L_ell = boussinesq.matrices(B,N,ell,Ekman,Prandtl,Rayleigh,alpha_BC,implicit_buoyancy=True,implicit_base_temp=True,boundary_condition=boundary_condition)

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

    return M_ell.astype(np.complex128), L_ell.astype(np.complex128), E_ell.astype(np.complex128)


def build_matrices_ell(B, Ekman, Prandtl, Rayleigh, ell_range, alpha_BC, boundary_condition):
    args = B, Ekman, Prandtl, Rayleigh, alpha_BC, boundary_condition
    args = [(ell, *args) for ell in ell_range]

    # Build matrices in parallel
    num_processes = min(mp.cpu_count(), 32)
    pool = mp.Pool(num_processes)
    result = pool.starmap(build_matrices_ell_fun, args)
    M, L, E = zip(*result)

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
        matsolver = 'SuperluColamdFactorized'
        evalues, evectors = scipy_sparse_eigs(Amat, Bmat, N=nev, target=lamtarget, matsolver=matsolver, profile=True)

    # Output data
    data = {'m': m, 'Lmax': B.L_max, 'Nmax': B.N_max,
            'boundary_condition': boundary_condition,
            'evalues': evalues, 'evectors': evectors,
            'Ekman': Ekman, 'Prandtl': Prandtl, 'Rayleigh': Rayleigh}
    filename = output_filename(m, B.L_max, B.N_max, boundary_condition, Ekman, Prandtl, Rayleigh, directory='data', ext='.pckl')
    save_data(filename, data)


def make_tensor_coeffs(m, L_max, N_max, R_max, rank, dtype='float64'):
    coeffs = []
    for ell in range(m, L_max+1):
        N = N_max+1 - ball128.N_min(ell-R_max)
        coeffs.append(np.zeros((3**rank*N,1), dtype=dtype))
    return coeffs


def expand_field(field, m, L_max, N_max, R_max, z, cos_theta):
    """Expand a field.  For now only supports scalar fields"""
    a = 0      # Default offset for Jacobi polynomials
    s = 0      # Spin weight
    alpha = 0  # How many times we've been differentiated
    R_max = 3  # Max rank

    Y = sphere128.Y(L_max, m, s, cos_theta).T

    f = 0*z[np.newaxis,:] + 0*cos_theta[:,np.newaxis]
    for ell in range(m, L_max+1):
        ell_local = ell-m
        N = N_max - ball128.N_min(ell-R_max)

        Qn = ball128.polynomial(N,alpha,ell,z,a=a)
        frad = field[ell_local].T @ Qn
        Yell = Y[:,ell_local][:,np.newaxis]
        f += frad * Yell

    return f


def plot_fields(fielddict, z, cos_theta):
    shading = 'gouraud'  # 'gouraud' interpolates but is slower than 'nearest'
    r = np.sqrt((z+1)/2)
    r, cos_theta = r[np.newaxis,:], cos_theta[:,np.newaxis]
    sin_theta = np.sqrt(1-cos_theta**2)
    x, z = r*sin_theta, r*cos_theta
    for name, field in fielddict.items():
        fig, ax = plt.subplots(1,1,figsize=(4.25,6))

        cmap = copy.copy(plt.get_cmap('RdBu'))
        cmap.set_bad(color='grey', alpha=.5)
        if shading == 'gouraud':
            # Shade via interpolation.  Can handle non-monotonic input grids
            c_im = ax.pcolormesh(x, z, field, cmap=cmap, shading='gouraud')
        else:
            # Nearest shading requires a monotonic input grid
            n = len(cos_theta)
            _    = ax.pcolormesh(x[:n,:], z[:n,:], field[:n,:], cmap=cmap, shading='nearest')
            c_im = ax.pcolormesh(x[n:,:], z[n:,:], field[n:,:], cmap=cmap, shading='nearest')
        cbar = fig.colorbar(c_im)
        eps = 0.02
        ax.plot((1+eps/2)*sin_theta, (1+eps/2)*cos_theta, color='k', linewidth=1)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(name)
        ax.set_aspect('equal')

        fig.show()


def plot_spectrum_callback(index, evalues, evectors, B, m, domain):
    plot_pressure = True
    plot_temperature = False
    plot_velocity = False
    evalue, evector = evalues[index], evectors[:,index]

    ntau = lambda ell: 1 if ell == 0 else 4
    state_vector = StateVector(B, 'mlr', [('u',1),('p',0),('T',0)], ntau=ntau, m_min=m, m_max=m)

    fielddict = {}
    maxreal, maximag = np.max(np.abs(evector.real)), np.max(np.abs(evector.imag))
    which = 'real' if maxreal > maximag else 'imag'
    print('Plotting {} part of eigenvector'.format(which))
    print('  imag/real ratio: {}'.format(maximag/maxreal))
    evector = 2 * (evector.real if which=='real' else evector.imag)

    nr, ntheta = 1024, 1025
    z, _ = ball128.quadrature(nr,a=0.0)
    cos_theta, _ = sphere128.quadrature(ntheta)

    # Collect fields, converting to grid space
    if plot_pressure:
        p = make_tensor_coeffs(m, B.L_max, B.N_max, B.R_max, rank=0)
        state_vector.unpack(evector, {'p': p})
        fielddict['p'] = expand_field(p, m, B.L_max, B.N_max, B.R_max, z, cos_theta)

    if plot_temperature:
        T = make_tensor_coeffs(m, B.L_max, B.N_max, B.R_max, rank=0)
        state_vector.unpack(evector, {'T': T})
        fielddict['T'] = expand_field(T, m, B.L_max, B.N_max, B.R_max, z, cos_theta)

    if plot_velocity:
        raise ValueError('Not implemented')
        u = make_tensor_coeffs(m, B.L_max, B.N_max, B.R_max, rank=1)
        state_vector.unpack(evector, {'u': u})
        fielddict['u'] = expand_field(u, m, B.L_max, B.N_max, B.R_max, z, cos_theta)

    # Plot
    plot_fields(fielddict, z, cos_theta)


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


def rotation_configs():
    return [{'Ekman': 10**-4,   'm': 6,  'omega': -.43346, 'Rayleigh': 5.1549, 'Lmax': 32,  'Nmax': 31},
            {'Ekman': 10**-4.5, 'm': 9,  'omega': -.44276, 'Rayleigh': 4.7613, 'Lmax': 32,  'Nmax': 31},
            {'Ekman': 10**-5,   'm': 14, 'omega': -.45715, 'Rayleigh': 4.5351, 'Lmax': 48,  'Nmax': 47},
            {'Ekman': 10**-5.5, 'm': 20, 'omega': -.45760, 'Rayleigh': 4.3937, 'Lmax': 92,  'Nmax': 63},
            {'Ekman': 10**-6,   'm': 30, 'omega': -.46394, 'Rayleigh': 4.3021, 'Lmax': 122, 'Nmax': 83},
            {'Ekman': 10**-6.5, 'm': 44, 'omega': -.46574, 'Rayleigh': 4.2416, 'Lmax': 152, 'Nmax': 101},
            {'Ekman': 10**-7,   'm': 65, 'omega': -.46803, 'Rayleigh': 4.2012, 'Lmax': 232, 'Nmax': 171},
            {'Ekman': 10**-7.5, 'm': 95, 'omega': -.46828, 'Rayleigh': 4.1742, 'Lmax': 296, 'Nmax': 201},
            {'Ekman': 10**-8,   'm': 139,'omega': -.43507, 'Rayleigh': 4.1527, 'Lmax': 500, 'Nmax': 500},
            {'Ekman': 10**-10,  'm': 646,'omega': -.43507, 'Rayleigh': 4.1527, 'Lmax': 1200, 'Nmax': 1200}
            ]


def compute_eigensolutions():
    solve = True
    plot = True

    boundary_condition = 'no-slip'
    nev = 10
    thermal_forcing_factor = 1

    configs = rotation_configs()
    configs = [configs[-1]]
    for config in configs:
        # Extract the domain parameters
        m, L_max, N_max = config['m'], config['Lmax'], config['Nmax']
        if L_max < m:
            raise ValueError('No angular resolution: L_max (={}) is too small'.format(L_max))

        B, domain = build_ball(m, L_max, N_max)

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
