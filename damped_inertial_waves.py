from dedalus_sphere import ball_wrapper as ball
from dedalus_sphere import ball128
import dedalus.public as de
from dedalus.core.distributor import Distributor
import numpy as np
import scipy.sparse as sparse
from mpi4py import MPI
import boussinesq
from state_vector import StateVector
from eigtools import eigsort, scipy_sparse_eigs
from plot_tools import dealias, plotequatorialslice, plotmeridionalslice, plotequatorialquiver, plotmeridionalquiver
import matplotlib.pyplot as plt


automatic_boundary_conditions = True

if automatic_boundary_conditions:
    def ntau(_):
        return 0
else:
    def ntau(ell):
        return 0 if ell == 0 else 3


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


def ell_matrices(B, N, ell, Ekman, alpha_BC, bc_type):
    """Construct grad.phi and boundary conditions for the inertial waves problem"""

    def D(mu, i, deg):
        if mu == +1: return B.op('D+', N, i, ell + deg)
        if mu == -1: return B.op('D-', N, i, ell + deg)

    def E(i, deg):
        return B.op('E', N, i, ell + deg)

    def C(deg): return ball128.connection(N,ell+deg,alpha_BC,2)

    I = B.op('I', N, 0, ell).tocsr()
    Z = B.op('0',N,0,ell)

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

    xim, xip = B.xi([-1,+1],ell)

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
        Q = B.Q[(ell,2)]
        if ell == 1: rDmm = 0.*B.op('r=1',N,1,ell)
        else: rDmm = B.xi(-1,ell-1)*B.op('r=1',N,1,ell-2)*D(-1,0,-1)
        rDpm = B.xi(+1,ell-1)*B.op('r=1',N,1,ell  )*D(+1,0,-1)
        rDm0 = B.xi(-1,ell  )*B.op('r=1',N,1,ell-1)*D(-1,0, 0)
        rDp0 = B.xi(+1,ell  )*B.op('r=1',N,1,ell+1)*D(+1,0, 0)
        rDmp = B.xi(-1,ell+1)*B.op('r=1',N,1,ell  )*D(-1,0,+1)
        rDpp = B.xi(+1,ell+1)*B.op('r=1',N,1,ell+2)*D(+1,0,+1)

        rD = np.array([rDmm, rDm0, rDmp, 0.*rDmm, 0.*rDm0, 0.*rDmp, rDpm, rDp0, rDpp])
        QSm = Q[:,::3].dot(rD[::3])
        QS0 = Q[:,1::3].dot(rD[1::3])
        QSp = Q[:,2::3].dot(rD[2::3])
        u0m = B.op('r=1',N,0,ell-1)*B.Q[(ell,1)][1,0]
        u0p = B.op('r=1',N,0,ell+1)*B.Q[(ell,1)][1,2]

        row0 = np.concatenate(( QSm[1]+QSm[3], QS0[1]+QS0[3] , QSp[1]+QSp[3], np.zeros(N3-N2)))
        row1 = np.concatenate(( u0m          , np.zeros(N0+1), u0p          , np.zeros(N3-N2)))
        row2 = np.concatenate(( QSm[5]+QSm[7], QS0[5]+QS0[7] , QSp[5]+QSp[7], np.zeros(N3-N2)))

    elif bc_type == 'no-slip':
        row0 = np.concatenate((                B.op('r=1', N, 0, ell-1), np.zeros(N3-N0)))
        row1 = np.concatenate((np.zeros(N0+1), B.op('r=1', N, 0, ell),   np.zeros(N3-N1)))
        row2 = np.concatenate((np.zeros(N1+1), B.op('r=1', N, 0, ell+1), np.zeros(N3-N2)))

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
        N = B.N_max - B.N_min(ell-B.R_max)
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


def damped_inertial_waves(B, m, domain, Ekman, boundary_condition):
    print('Computing Inertial Wave Solutions...')

    alpha_BC = 0
    save_plots = False
    solve_dense = True
    target_id = (4,1,0)

    targets = {}  # Greenspan targets, final index == m
    targets[(2,1,1)] = 1
    targets[(4,1,0)] = 1.309
    targets[(4,1,1)] = 0.820
    targets[(4,2,1)] = 1.708
    targets[(4,3,1)] = 0.612
    targets[(6,1,0)] = 0.938
    targets[(6,2,0)] = 1.660
    targets[(8,1,0)] = 0.726
    targets[(8,2,0)] = 1.354
    targets[(8,3,0)] = 1.800

    if save_plots:
        def savefig(fn): plt.savefig(fn)
    else:
        def savefig(_): pass

    # Construct the state vector
    state_vector = StateVector(B, 'mlr', [('u',1),('p',0)], ntau=ntau, m_min=m, m_max=m)

    # Build up the linear system
    print('  Building matrices...')
    Amat, Bmat = build_matrices(B, state_vector, m, Ekman=Ekman, alpha_BC=alpha_BC, bc_type=boundary_condition)

    if solve_dense:
        # Compute eigenvalues
        print('  Solving eigenproblem for m = {}, size {}x{}'.format(m, np.shape(Amat)[0], np.shape(Amat)[1]))
        lam, v = eigsort(Amat.todense(), Bmat.todense(), cutoff=1e9)

        plt.plot(np.real(lam), np.imag(lam), '.', markersize=3, color='tab:blue')
        plt.xlabel('Real(λ)')
        plt.ylabel('Imag(λ)')
        plt.title('Damped Inertial Wave Eigenvalues, Ekman = {:1.6e}, m = {}, Lmax = {}'.format(Ekman, m, B.L_max))
        plt.grid(True)
        filename = 'figures/damped_inertial_waves/damped_inertial_wave_eigenvalues-Ekman={:1.6e}-m={}-Lmax={}'.format(Ekman, m, B.L_max)
        savefig(filename + '.png')

        plt.show()
    else:
        print('  Solving sparse eigenproblem for m = {}, L_max = {}, N_max = {}, size {}x{}'.format(
                m, B.L_max, B.N_max, np.shape(Amat)[0], np.shape(Amat)[1]),  flush=True)
        lam, v = scipy_sparse_eigs(Amat, Bmat, N=1, target=1j*targets[target_id], profile=True)

    target_ids = [target_id]
    for index in target_ids:
        modestr = str(index[0]) + str(index[1]) + str(index[2])

        eval_target = 1j*targets[index]
        ind = np.argmin(abs(lam - eval_target))
        eval = lam[ind]
        evec = v[:, ind]

        print("  Plotting eigenvectors for Greenspan Mode {}, λ = {:1.5e}...".format(index, eval))
        # Unpack the eigenvector into our tensor fields
        u = ball.TensorField_3D(1, B, domain)
        p = ball.TensorField_3D(0, B, domain)
        state_vector.unpack(evec, [u, p])

        # Dealias for plotting
        res = 256
        L_factor, N_factor = res // (B.L_max + 1), res // (B.N_max + 1)
        p, _, _, _ = dealias(B, domain, p, L_factor=L_factor, N_factor=N_factor)
        u, r, theta, phi = dealias(B, domain, u, L_factor=L_factor, N_factor=N_factor)

        p = p['g'][0]
        ur, utheta, uphi = u['g'][0], u['g'][1], u['g'][2]

        filename = lambda field, sl: 'figures/damped_inertial_waves/damped_inertial_wave-Ekman={:1.6e}-mode={}-bc={}-field={}-slice={}.png'.format(Ekman, modestr, boundary_condition, field, sl)

        # Plot
        if m > 0:
            plotequatorialslice(ur, r, theta, phi)
            plt.title('Ekman = {:1.4e}, λ = {:1.4e}, Equatorial Slice, $u_r$'.format(Ekman, eval))
            savefig(filename('ur', 'e'))

            plotequatorialslice(utheta, r, theta, phi)
            plt.title('Ekman = {:1.4e}, λ = {:1.4e}, Equatorial Slice, $u_Θ$'.format(Ekman, eval))
            savefig(filename('utheta', 'e'))

            plotequatorialslice(uphi, r, theta, phi)
            plt.title('Ekman = {:1.4e}, λ = {:1.4e}, Equatorial Slice, $u_ϕ$'.format(Ekman, eval))
            savefig(filename('uphi', 'e'))

        angle = 0.

        plotmeridionalslice(p, r, theta, phi, angle=angle)
        plt.title('Ekman = {:1.4e}, λ = {:1.4e}, pressure'.format(Ekman, eval))
        savefig(filename('p', 'm'))

        plotmeridionalslice(ur, r, theta, phi, angle=angle)
        plt.title('Ekman = {:1.4e}, λ = {:1.4e}, $u_r$'.format(Ekman, eval))
        savefig(filename('ur', 'm'))

        plotmeridionalslice(uphi, r, theta, phi, angle=angle)
        plt.title('Ekman = {:1.4e}, λ = {:1.4e}, $u_φ$'.format(Ekman, eval))
        savefig(filename('uphi', 'm'))

        plotmeridionalslice(utheta, r, theta, phi, angle=angle)
        plt.title('Ekman = {:1.4e}, λ = {:1.4e}, $u_Θ$'.format(Ekman, eval))
        savefig(filename('utheta', 'm'))

        # Normalized kinetic energy
        ke = np.log10(0.5 * (u['g'][0] ** 2 + u['g'][1] ** 2 + u['g'][2] ** 2))
        ke -= np.max(ke)
        truncate_level = -8
        ketrunc = np.where(ke < truncate_level, np.nan, ke)

        plotmeridionalslice(ketrunc, r, theta, phi, angle=angle, cmap='RdBu_r')
        plt.title('Ekman = {:1.4e}, λ = {:1.4e}, Kinetic Energy'.format(Ekman, eval))
        savefig(filename('ke', 'm'))


def main():
    # Create the domain
    m = 0
    L_max, N_max = 15, 15
    Ekman = 10**-1
    # boundary_condition = 'stress-free'
    boundary_condition = 'no-slip'

    B, domain = build_ball(L_max=L_max, N_max=N_max)
    damped_inertial_waves(B, m, domain, Ekman, boundary_condition)
    plt.show()


if __name__=='__main__':
    main()

