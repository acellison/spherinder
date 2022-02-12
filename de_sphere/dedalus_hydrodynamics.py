from dedalus_sphere import ball_wrapper as ball
from dedalus_sphere import ball128, sphere128
import dedalus.public as de

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

import boussinesq
from state_vector import StateVector
from plot_tools import dealias, plotmeridionalslice

from utilities import build_ball, make_tensor_coeffs, expand_field, plot_fields
from utilities import save_data, save_figure

from spherinder.eigtools import eigsort, plot_spectrum, scipy_sparse_eigs


g_file_prefix = 'dedalus_hydrodynamics'
automatic_boundary_conditions = False

if automatic_boundary_conditions:
    def ntau(_):
        return 0
else:
    def ntau(ell):
        return 0 if ell == 0 else 3


def ell_matrices(B, N, ell, Ekman, alpha_BC, bc_type):
    """Construct grad.phi and boundary conditions for the inertial waves problem"""
    def op(op_name,N,k,ell,a=B.a,dtype=np.float64):
        return ball128.operator(op_name,N,k,ell,a=a).astype(dtype)

    def xi(mu,ell):
        # returns xi for ell > 0 or ell = 0 and mu = +1
        # otherwise returns 0.
        if (ell > 0) or (ell == 0 and mu == 1):
            return ball128.xi(mu,ell)
        return 0.

    def D(mu, i, deg):
        if mu == +1: return op('D+', N, i, ell + deg)
        if mu == -1: return op('D-', N, i, ell + deg)

    def E(i, deg):
        return op('E', N, i, ell + deg)

    def C(deg): return ball128.connection(N,ell+deg,alpha_BC,2)

    I = op('I', N, 0, ell).tocsr()
    Z = op('0',N,0,ell)

    N0 = N
    N1 = N + N0 + 1
    N2 = N + N1 + 1
    N3 = N + N2 + 1

    if ell == 0:
        # Set u, p == 0
        M = sparse.bmat([[Z,Z,Z,Z],
                         [Z,Z,Z,Z],
                         [Z,Z,Z,Z],
                         [Z,Z,Z,Z]]).tocsr()
        L = sparse.bmat([[I,Z,Z,Z],
                         [Z,I,Z,Z],
                         [Z,Z,I,Z],
                         [Z,Z,Z,I]]).tocsr()
        return M, L, L

    xim, xip = xi([-1,+1],ell)

    M00 = E(1,-1).dot(E(0,-1))
    M11 = E(1, 0).dot(E(0, 0))
    M22 = E(1,+1).dot(E(0,+1))

    M = sparse.bmat([[M00,   Z,   Z,  Z],
                     [  Z, M11,   Z,  Z],
                     [  Z,   Z, M22,  Z],
                     [  Z,   Z,   Z,  Z]]).tocsr()

    L00 = -Ekman*D(-1, 1, 0).dot(D(+1, 0,-1))
    L11 = -Ekman*D(-1, 1,+1).dot(D(+1, 0, 0))
    L22 = -Ekman*D(+1, 1, 0).dot(D(-1, 0,+1))

    L03 = xim*E(+1,-1).dot(D(-1,0,0))
    L23 = xip*E(+1,+1).dot(D(+1,0,0))

    L30 = xim*D(+1,0,-1)
    L32 = xip*D(-1,0,+1)

    L = sparse.bmat([[L00, Z,   Z,   L03],
                     [Z,   L11, Z,   Z  ],
                     [Z,   Z,   L22, L23],
                     [L30, Z,   L32, Z  ]]).tocsr()

    if automatic_boundary_conditions:
        # We can multiply by 1-r**2 by multiplying on the right by C_{alpha,ell}^T
        # This will yield no-slip boundary conditions for u
        Q = sparse.block_diag([E(0,-1).T, E(0,0).T, E(0,+1).T, I]).tocsr()
        return M, L, Q

    if bc_type == 'stress-free':
        Q = boussinesq.make_Q([ell],2)
        if ell == 1: rDmm = 0.*op('r=1',N,1,ell)
        else: rDmm = xi(-1,ell-1)*op('r=1',N,1,ell-2)*D(-1,0,-1)
        rDpm = xi(+1,ell-1)*op('r=1',N,1,ell  )*D(+1,0,-1)
        rDm0 = xi(-1,ell  )*op('r=1',N,1,ell-1)*D(-1,0, 0)
        rDp0 = xi(+1,ell  )*op('r=1',N,1,ell+1)*D(+1,0, 0)
        rDmp = xi(-1,ell+1)*op('r=1',N,1,ell  )*D(-1,0,+1)
        rDpp = xi(+1,ell+1)*op('r=1',N,1,ell+2)*D(+1,0,+1)

        rD = np.array([rDmm, rDm0, rDmp, 0.*rDmm, 0.*rDm0, 0.*rDmp, rDpm, rDp0, rDpp])
        QSm = Q[:,::3].dot(rD[::3])
        QS0 = Q[:,1::3].dot(rD[1::3])
        QSp = Q[:,2::3].dot(rD[2::3])
        u0m = op('r=1',N,0,ell-1)*B.Q[(ell,1)][1,0]
        u0p = op('r=1',N,0,ell+1)*B.Q[(ell,1)][1,2]

        row0 = np.concatenate(( QSm[1]+QSm[3], QS0[1]+QS0[3] , QSp[1]+QSp[3], np.zeros(N3-N2)))
        row1 = np.concatenate(( u0m          , np.zeros(N0+1), u0p          , np.zeros(N3-N2)))
        row2 = np.concatenate(( QSm[5]+QSm[7], QS0[5]+QS0[7] , QSp[5]+QSp[7], np.zeros(N3-N2)))

    elif bc_type == 'no-slip':
        row0 = np.concatenate((                op('r=1', N, 0, ell-1), np.zeros(N3-N0)))
        row1 = np.concatenate((np.zeros(N0+1), op('r=1', N, 0, ell),   np.zeros(N3-N1)))
        row2 = np.concatenate((np.zeros(N1+1), op('r=1', N, 0, ell+1), np.zeros(N3-N2)))

    else:
        raise ValueError("bc_type must be one of 'no-slip' or 'stress-free'")

    tau0 = C(-1)[:,-1]
    tau1 = C( 0)[:,-1]
    tau2 = C(+1)[:,-1]

    tau0 = tau0.reshape((len(tau0),1))
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = tau2.reshape((len(tau2),1))

    col0 = np.concatenate((                   tau0,np.zeros((N3-N0,1))))
    col1 = np.concatenate((np.zeros((N0+1,1)),tau1,np.zeros((N3-N1,1))))
    col2 = np.concatenate((np.zeros((N1+1,1)),tau2,np.zeros((N3-N2,1))))

    L = sparse.bmat([[   L, col0, col1, col2],
                     [row0,    0 ,   0,    0],
                     [row1,    0 ,   0,    0],
                     [row2,    0,    0,    0]])

    M = sparse.bmat([[     M, 0*col0, 0*col1, 0*col2],
                     [0*row0,      0,      0,      0],
                     [0*row1,      0,      0,      0],
                     [0*row2,      0,      0,      0]])

    L = L.tocsr()
    M = M.tocsr()
    return M, L, None


def build_matrices(B, state_vector, m, Ekman, alpha_BC, bc_type):
    ell_range = range(m, B.L_max+1)
    Mmats, Lmats, Qmats = [], [], []
    for ell in ell_range:
        N = B.N_max - ball128.N_min(ell-B.R_max)
        M_ell, L_ell, Q_ell = ell_matrices(B, N, ell, Ekman, alpha_BC, bc_type)
        Mmats.append(M_ell.astype(np.complex128))
        Lmats.append(L_ell.astype(np.complex128))
        Qmats.append(Q_ell)

    # Build the block matrices for fixed m
    M = sparse.block_diag([M_ell for M_ell in Mmats])
    L = sparse.block_diag([L_ell for L_ell in Lmats])

    # Create the Coriolis matrix
    C = boussinesq.create_coriolis_matrix(B, state_vector, m_min=m, m_max=m)

    # Convert Cm to alpha = 2
    C = M @ C

    # Automatic boundary conditions
    if automatic_boundary_conditions:
        Q = sparse.block_diag([Q_ell for Q_ell in Qmats])
        M, L, C = M @ Q, L @ Q, C @ Q

    Amat = L + 2*C
    Bmat = -M
    return Amat, Bmat


def make_filename_prefix(directory='data'):
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), directory))
    abspath = os.path.join(basepath, g_file_prefix)
    return os.path.join(abspath, g_file_prefix)


def output_filename(m, Lmax, Nmax, boundary_condition, Ekman, directory, ext, prefix='evalues'):
    return make_filename_prefix(directory) + f'-{prefix}-m={m}-Lmax={Lmax}-Nmax={Nmax}-Ekman={Ekman:1.4e}-{boundary_condition}' + ext


def solve_eigenproblem(B, m, domain, Ekman, boundary_condition, nev, evalue_target):
    print('Computing Hydrodynamics Solutions...')

    alpha_BC = 0

    # Construct the state vector
    state_vector = StateVector(B, 'mlr', [('u',1),('p',0)], ntau=ntau, m_min=m, m_max=m)

    # Build up the linear system
    print('  Building matrices...')
    Amat, Bmat = build_matrices(B, state_vector, m, Ekman=Ekman, alpha_BC=alpha_BC, bc_type=boundary_condition)

    # Compute eigenvalues
    print('  Solving eigenproblem for m = {}, size {}x{}'.format(m, np.shape(Amat)[0], np.shape(Amat)[1]))
    if nev == 'all':
        evalues, evectors = eigsort(Amat.todense(), Bmat.todense(), cutoff=1e9)
    else:
        matsolver = 'SuperluColamdFactorized'
        evalues, evectors = scipy_sparse_eigs(Amat, Bmat, N=nev, target=evalue_target, matsolver=matsolver, profile=True)

    # Output data
    data = {'m': m, 'Lmax': B.L_max, 'Nmax': B.N_max, 
            'boundary_condition': boundary_condition,
            'evalues': evalues, 'evectors': evectors,
            'Ekman': Ekman}
    filename = output_filename(m, B.L_max, B.N_max, boundary_condition, Ekman, directory='data', ext='.pckl')
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
        fielddict['p'] = pfield.real if np.max(abs(pfield.real)) > np.max(abs(pfield.imag)) else pfield.imag

    if plot_velocity:
        raise ValueError('Not implemented')
        u = make_tensor_coeffs(m, B.L_max, B.N_max, B.R_max, rank=1)
        state_vector.unpack(evector, {'u': u})
        fielddict['u'] = expand_field(u, m, B.L_max, B.N_max, B.R_max, z, cos_theta)

    # Plot
    plot_fields(fielddict, z, cos_theta, colorbar=False)


def plot_solution(B, m, domain, Ekman, boundary_condition):
    # Load the data
    filename = output_filename(m, B.L_max, B.N_max, boundary_condition, Ekman, directory='data', ext='.pckl')
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    plot_fields = True
    if plot_fields:
        onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, B, m, domain)
    else:
        onpick = None

    fig, ax = plot_spectrum(evalues, onpick)
    ax.set_xlim([-.1525, -0.0025])
    ax.set_ylim([-.11, .21])
    ax.set_title('Dedalus Sphere Basis')

    plot_filename = output_filename(m, B.L_max, B.N_max, boundary_condition, Ekman, directory='figures', ext='.png')
    save_figure(plot_filename, fig)

    fig.show()


def main():
    solve = True
    plot = True

    # Create the domain
#    m, Ekman, L_max, N_max, nev, evalue_target = 14, 1e-5, 40, 40, 'all', None
    m, Ekman, L_max, N_max, nev, evalue_target = 30, 1e-6, 280, 146, 1000, -0.0070738+0.060679j

    # boundary_condition = 'stress-free'
    boundary_condition = 'no-slip'

    B, domain = build_ball(m, L_max, N_max)

    if solve:
        solve_eigenproblem(B, m, domain, Ekman, boundary_condition, nev, evalue_target)

    if plot:
        plot_solution(B, m, domain, Ekman, boundary_condition)
        plt.show()


if __name__=='__main__':
    main()

