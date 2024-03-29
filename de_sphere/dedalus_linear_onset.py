import pickle, os
import time
import multiprocessing as mp

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from dedalus_sphere import ball128, sphere128
import boussinesq
from state_vector import StateVector

from spherinder.eigtools import scipy_sparse_eigs, eigsort, plot_spectrum

from utilities import build_ball, make_tensor_coeffs, expand_field, plot_fields
from utilities import save_data, save_figure


g_file_prefix = 'dedalus_linear_onset'


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
    parallel = False
    if parallel:
        num_processes = min(mp.cpu_count(), 32)
        pool = mp.Pool(num_processes)
        result = pool.starmap(build_matrices_ell_fun, args)
    else:
        result = [build_matrices_ell_fun(*a) for a in args]
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

        print('Most critical eigenvalue: ', evalues[-1])
        print(f'                   omega: {evalues[-1].imag*Ekman**(2/3):1.5f}')

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

    print('Most critical eigenvalue: ', evalues[-1])
    print(f'                   omega: {evalues[-1].imag*Ekman**(2/3):1.5f}')
    return

    plot_fields = False
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
    """No-slip critical configurations"""
    return [{'Ekman': 10**-4,   'm': 6,  'omega': -.27009, 'Rayleigh': 5.0151,  'Lmax': 32,  'Nmax': 31},
            {'Ekman': 10**-4.5, 'm': 9,  'omega': -.31035, 'Rayleigh': 4.6581,  'Lmax': 32,  'Nmax': 31},
            {'Ekman': 10**-5,   'm': 13, 'omega': -.33901, 'Rayleigh': 4.4660,  'Lmax': 48,  'Nmax': 47},
            {'Ekman': 10**-5.5, 'm': 20, 'omega': -.36778, 'Rayleigh': 4.3488,  'Lmax': 92,  'Nmax': 63},
            {'Ekman': 10**-6,   'm': 30, 'omega': -.38930, 'Rayleigh': 4.2736,  'Lmax': 122, 'Nmax': 83},
            {'Ekman': 10**-6.5, 'm': 44, 'omega': -.40439, 'Rayleigh': 4.22355, 'Lmax': 152, 'Nmax': 101},
            {'Ekman': 10**-7,   'm': 65, 'omega': -.41737, 'Rayleigh': 4.19025, 'Lmax': 232, 'Nmax': 171},
            {'Ekman': 10**-7.5, 'm': 95, 'omega': -.42658, 'Rayleigh': 4.16773, 'Lmax': 320, 'Nmax': 260},
            {'Ekman': 10**-8,   'm': 139,'omega': -.43406, 'Rayleigh': 4.1527,  'Lmax': 400, 'Nmax': 340},   # NOT VERIFIED
            ]

def _solve_helper(config_index):
    boundary_condition = 'no-slip'
    thermal_forcing_factor = 1
    nev = 10

    config = rotation_configs()[config_index]
    m, L_max, N_max = config['m'], config['Lmax'], config['Nmax']
    if L_max < m:
        raise ValueError('No angular resolution: L_max (={}) is too small'.format(L_max))
    Ekman, Prandtl, Rayleigh = config['Ekman'], 1, config['Rayleigh']

    B, domain = build_ball(m, L_max, N_max)

    # Skip if we already have it
    filename = output_filename(m, B.L_max, B.N_max, boundary_condition, Ekman, Prandtl, Rayleigh, directory='data', ext='.pckl')
    if os.path.exists(filename):
        print('  Already solved')
        return

    solve_eigenproblem(B, m, domain, config, nev=nev, boundary_condition=boundary_condition,
                       thermal_forcing_factor=thermal_forcing_factor)


def solve():
    indices = range(8)

    pool = mp.Pool(mp.cpu_count()-1)
    pool.map(_solve_helper, indices)


def plot_modes():
    nrows, ncols = 2, 4
    configs = rotation_configs()[:nrows*ncols]
    boundary_condition = 'no-slip'
    plot_pressure = True

    fig, plot_axes = plt.subplots(nrows,ncols,figsize=(ncols*3,nrows*5.25))

    for index, config in enumerate(configs):
        # Extract the domain parameters
        m, L_max, N_max = config['m'], config['Lmax'], config['Nmax']
        B, domain = build_ball(m, L_max, N_max)

        # Get reduced nondimensional parameters from config
        Ekman, Rayleigh, omega = config['Ekman'], config['Rayleigh'], config['omega']

        # Rescale parameters
        omega /= Ekman**(2/3)
        Prandtl = 1
        thermal_forcing_factor = 1
        Rayleigh = thermal_forcing_factor * Rayleigh / Ekman**(1/3)

        # Load the data
        filename = output_filename(m, B.L_max, B.N_max, boundary_condition, Ekman, Prandtl, Rayleigh, directory='data', ext='.pckl')
        data = pickle.load(open(filename, 'rb'))

        # Extract configuration parameters
        evalues, evectors = data['evalues'], data['evectors']
        evalue, evector = evalues[-1], evectors[:,-1]
        print(f'm = {m}, Ekamn = {Ekman:1.3e}, Critical Eigenvalue: {evalue}')

        ntau = lambda ell: 1 if ell == 0 else 4
        state_vector = StateVector(B, 'mlr', [('u',1),('p',0),('T',0)], ntau=ntau, m_min=m, m_max=m)

        fielddict = {}
        maxreal, maximag = np.max(np.abs(evector.real)), np.max(np.abs(evector.imag))
        which = 'real' if maxreal > maximag else 'imag'
        evector = 2 * (evector.real if which=='real' else evector.imag)

        nr, ntheta = 1024, 1025
        z, _ = ball128.quadrature(nr,a=0.0)
        cos_theta, _ = sphere128.quadrature(ntheta)

        # Collect fields, converting to grid space
        name = 'p' if plot_pressure else 'T'
        coeffs = make_tensor_coeffs(m, B.L_max, B.N_max, B.R_max, rank=0)
        state_vector.unpack(evector, {name: coeffs})
        field = expand_field(coeffs, m, B.L_max, B.N_max, B.R_max, z, cos_theta)

        # Orient them all the same way
        if abs(np.min(field)) > np.max(field):
            field = -field

        # Plot
        row, col = index//ncols, index%ncols
        ax = plot_axes[row][col]
        plot_fields({name: field}, z, cos_theta, colorbar=False, fig=fig, ax=ax, cmap=plt.get_cmap('RdBu_r'))

        if col != 0:
            ax.set_ylabel(None)
            ax.set_yticklabels([])
        if row != nrows-1:
            ax.set_xlabel(None)
            ax.set_xticklabels([])

        logek = np.log10(Ekman)
        ax.set_title('E = $10^{' + f'{logek:1.1f}' + '}$')

    fig.set_tight_layout(True)

    # Save the figure
    prefix = make_filename_prefix('figures')
    field = 'pressure' if plot_pressure else 'temperature'
    plot_filename = prefix + f'-critical_modes-{field}.png'
    save_figure(plot_filename, fig)

    plt.show()


def main():
    solve = True
    plot = False

    boundary_condition = 'no-slip'
    nev = 3
    thermal_forcing_factor = 1

    config_index = 7
    resolution_ratio = 1.0
    delta_m = 0

    configs = [rotation_configs()[config_index]]
    for config in configs:
        # Extract the domain parameters
        m, L_max, N_max = config['m'], config['Lmax'], config['Nmax']
        m = m+delta_m
        L_max, N_max = [int(resolution_ratio*a) for a in [L_max, N_max]]
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
    main()
#    solve()
#    plot_modes()

