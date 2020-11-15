import os, pickle
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from dedalus_sphere import ball128, sphere128

import boussinesq
from state_vector import StateVector

from spherinder.eigtools import eigsort, plot_spectrum

from utilities import build_ball, make_tensor_coeffs, expand_field, plot_fields
from utilities import save_data, save_figure


g_file_prefix = 'dedalus_inertial_waves'


def ntau(ell):
    return 0 if ell == 0 else 1


def ell_matrices(B, N, ell, alpha_BC):
    """Construct grad.phi and boundary conditions for the inertial waves problem"""
    def op(op_name,N,k,ell,a=B.a,dtype=np.float64):
        return ball128.operator(op_name,N,k,ell,a=a).astype(dtype)

    def xi(mu,ell):
        # returns xi for ell > 0 or ell = 0 and mu = +1
        # otherwise returns 0.
        if (ell > 0) or (ell == 0 and mu == 1):
            return ball128.xi(mu,ell)
        return 0.

    def D(mu,i,deg):
        if mu == +1: return op('D+',N,i,ell+deg)
        if mu == -1: return op('D-',N,i,ell+deg)

    def C(alpha,deg): return ball128.connection(N,ell+deg,alpha_BC,alpha)

    Z = op('0',N,0,ell)
    I = op('I',N,0,ell)

    N0 = N
    N1 = N + N0 + 1
    N2 = N + N1 + 1
    N3 = N + N2 + 1

    if ell == 0:
        # Set u, p == 0
        I = op('I',N,0,ell).tocsr()
        M = sparse.bmat([[Z,Z,Z,Z],
                         [Z,Z,Z,Z],
                         [Z,Z,Z,Z],
                         [Z,Z,Z,Z]]).tocsr()
        L = sparse.bmat([[I,Z,Z,Z],
                         [Z,I,Z,Z],
                         [Z,Z,I,Z],
                         [Z,Z,Z,I]]).tocsr()
        E = sparse.block_diag([Z, Z, Z, Z])
        return M, L, E

    xim, xip = xi([-1,+1],ell)

    M00 = op('E',N,0,ell-1)
    M11 = I
    M22 = op('E',N,0,ell+1)

    M=sparse.bmat([[M00,   Z,   Z,  Z],
                   [  Z, M11,   Z,  Z],
                   [  Z,   Z, M22,  Z],
                   [  Z,   Z,   Z,  Z]]).tocsr()

    L03 = xim*D(-1,0,0)
    L23 = xip*D(+1,0,0)

    L30 = xim*D(+1,0,-1)
    L32 = xip*D(-1,0,+1)

    L = sparse.bmat([[Z,   Z, Z,   L03],
                     [Z,   Z, Z,   Z  ],
                     [Z,   Z, Z,   L23],
                     [L30, Z, L32, Z  ]]).tocsr()

    # u.r = 0 on boundary
    Q = boussinesq.make_Q([ell],1)
    u0m = op('r=1',N,0,ell-1)*Q[(ell,1)][1,0]
    u0p = op('r=1',N,0,ell+1)*Q[(ell,1)][1,2]
    row = np.concatenate((u0m, np.zeros(N0+1), u0p, np.zeros(N3-N2)))

    tau0 = C(1,-1)[:,-1]
    tau1 = C(0, 0)[:,-1]
    tau2 = C(1, 1)[:,-1]
    tau3 = C(1, 0)[:,-1]

    tau0 = tau0.reshape((len(tau0),1))
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = tau2.reshape((len(tau2),1))
    tau3 = tau3.reshape((len(tau3),1))

    col0 = np.concatenate((                   tau0,np.zeros((N3-N0,1))))
    col1 = np.concatenate((np.zeros((N0+1,1)),tau1,np.zeros((N3-N1,1))))
    col2 = np.concatenate((np.zeros((N1+1,1)),tau2,np.zeros((N3-N2,1))))
    col3 = np.concatenate((np.zeros((N2+1,1)),tau3))
    cols = [col0, col1, col2, col3]

    whichtau = 3  # Selects the equation to take the tau error
    col = cols[whichtau]

    L = sparse.bmat([[   L, col],
                     [row,    0]])

    M = sparse.bmat([[    M, 0*col],
                     [0*row,     0]])

    # Conversion matrices for Coriolis
    E00 = op('E',N,0,ell-1)
    E11 = I
    E22 = op('E',N,0,ell+1)
    E = sparse.block_diag([E00, E11, E22, Z])

    # Append the tau rows and columns
    nr, nc = np.shape(E)
    nt = ntau(ell)
    col = np.zeros((nr, nt))
    row = np.zeros((nt, nc))
    E = sparse.bmat([[E, col], [row, np.zeros((nt, nt))]])

    L = L.tocsr()
    M = M.tocsr()
    E = E.tocsr()

    return M, L, E


def build_matrices(B, state_vector, m, alpha_BC):
    ell_range = range(m, B.L_max+1)
    M, L, E = [], [], []
    for ell in ell_range:
        N = B.N_max - ball128.N_min(ell-B.R_max)
        M_ell, L_ell, E_ell = ell_matrices(B, N, ell, alpha_BC)
        M.append(M_ell.astype(np.complex128))
        L.append(L_ell.astype(np.complex128))
        E.append(E_ell.astype(np.complex128))

    # Build the block matrices for fixed m
    Mm = sparse.block_diag([M_ell for M_ell in M])
    Lm = sparse.block_diag([L_ell for L_ell in L])
    Em = sparse.block_diag([E_ell for E_ell in E])

    # Create the Coriolis matrix
    Cm = boussinesq.create_coriolis_matrix(B, state_vector, m_min=m, m_max=m)

    # Convert Cm to alpha = 1 (where appropriate)
    Cm = Em.dot(Cm)

    Amat = Lm + 2 * Cm
    Bmat = -1j * Mm
    return Amat, Bmat


def make_filename_prefix(directory='data'):
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), directory))
    abspath = os.path.join(basepath, g_file_prefix)
    return os.path.join(abspath, g_file_prefix)


def output_filename(m, Lmax, Nmax, directory, ext, prefix='evalues'):
    return make_filename_prefix(directory) + f'-{prefix}-m={m}-Lmax={Lmax}-Nmax={Nmax}' + ext


def solve_eigenproblem(B, m, domain):
    print(f'Computing Inertial Wave Solutions, m = {m}, L_max = {B.L_max}, N_max = {B.N_max}')
    alpha_BC = 0

    # Construct the state vector
    state_vector = StateVector(B, 'mlr', [('u',1),('p',0)], ntau=ntau, m_min=m, m_max=m)

    # Build up the linear system
    print('  Building matrices')
    Amat, Bmat = build_matrices(B, state_vector, m, alpha_BC=alpha_BC)

    # Compute eigenvalues
    print('  Solving eigenproblem for m = {}, L_max = {}, N_max = {}, size {}x{}'.format(
        m, B.L_max, B.N_max, np.shape(Amat)[0], np.shape(Amat)[1]))
    evalues, evectors = eigsort(Amat.todense(), Bmat.todense())

    # Output data
    data = {'m': m, 'Lmax': B.L_max, 'Nmax': B.N_max,
            'evalues': evalues, 'evectors': evectors}
    filename = output_filename(m, B.L_max, B.N_max, directory='data', ext='.pckl')
    save_data(filename, data)


def plot_spectrum_callback(index, evalues, evectors, B, m, domain):
    plot_pressure = True
    plot_velocity = False
    evalue, evector = evalues[index], evectors[:,index]

    state_vector = StateVector(B, 'mlr', [('u',1),('p',0)], ntau=ntau, m_min=m, m_max=m)

    maxreal, maximag = np.max(np.abs(evector.real)), np.max(np.abs(evector.imag))
    which = 'real' if maxreal > maximag else 'imag'
    print('Plotting {} part of eigenvector'.format(which))
    print('  imag/real ratio: {}'.format(maximag/maxreal))

    nr, ntheta = 1024, 1025
    z, _ = ball128.quadrature(nr,a=0.0)
    cos_theta, _ = sphere128.quadrature(ntheta)

    # Collect fields, converting to grid space
    fielddict = {}
    if plot_pressure:
        p = make_tensor_coeffs(m, B.L_max, B.N_max, B.R_max, rank=0, dtype='complex128')
        state_vector.unpack(evector, {'p': p})
        pfield = expand_field(p, m, B.L_max, B.N_max, B.R_max, z, cos_theta)
        fielddict['p'] = pfield.real if which == 'imag' else pfield.imag  # flip sense

    if plot_velocity:
        raise ValueError('Not implemented')
        u = make_tensor_coeffs(m, B.L_max, B.N_max, B.R_max, rank=1)
        state_vector.unpack(evector, {'u': u})
        fielddict['u'] = expand_field(u, m, B.L_max, B.N_max, B.R_max, z, cos_theta)

    # Plot
    plot_fields(fielddict, z, cos_theta, colorbar=False)


def plot_solution(B, m, domain):
    print('Plotting solution')
    # Load the data
    filename = output_filename(m, B.L_max, B.N_max, directory='data', ext='.pckl')
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    plot_fields = True
    if plot_fields:
        onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, B, m, domain)
    else:
        onpick = None

    fig, ax = plot_spectrum(evalues, onpick)
    ax.set_xlim([-2.1,2.1])
    ax.set_title('Inertial Wave Eigenvalues')
    ax.set_xlabel('Real(λ)')
    ax.set_ylabel('Imag(λ)')

    plot_filename = output_filename(m, B.L_max, B.N_max, directory='figures', ext='.png')
    save_figure(plot_filename, fig)

    fig.show()


def main():
    solve = True
    plot = True

    # Create the domain
    m = 139
    L_max = m+81
    N_max = 110

    # Extract the domain parameters
    if L_max < m:
        raise ValueError('No angular resolution: L_max (={}) is too small'.format(L_max))

    B, domain = build_ball(m, L_max, N_max)

    # Solve the eigenproblem
    if solve:
        solve_eigenproblem(B, m, domain)

    # Plot the solution
    if plot:
        plot_solution(B, m, domain)

    plt.show()



if __name__=='__main__':
    main()

