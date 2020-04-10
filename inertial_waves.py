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


def ntau(ell):
    return 0 if ell == 0 else 1
    # return 0 if ell == 0 else 3


def ell_matrices(B, N, ell, alpha_BC):
    """Construct grad.phi and boundary conditions for the inertial waves problem"""
    def D(mu,i,deg):
        if mu == +1: return B.op('D+',N,i,ell+deg)
        if mu == -1: return B.op('D-',N,i,ell+deg)

    def C(alpha,deg): return ball128.connection(N,ell+deg,alpha_BC,alpha)

    Z = B.op('0',N,0,ell)
    I = B.op('I',N,0,ell)

    N0 = N
    N1 = N + N0 + 1
    N2 = N + N1 + 1
    N3 = N + N2 + 1

    if ell == 0:
        # Set u, p == 0
        I = B.op('I',N,0,ell).tocsr()
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

    xim, xip = B.xi([-1,+1],ell)

    M00 = B.op('E',N,0,ell-1)
    M11 = I
    M22 = B.op('E',N,0,ell+1)

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
    u0m = B.op('r=1',N,0,ell-1)*B.Q[(ell,1)][1,0]
    u0p = B.op('r=1',N,0,ell+1)*B.Q[(ell,1)][1,2]
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
    E00 = B.op('E',N,0,ell-1)
    E11 = I
    E22 = B.op('E',N,0,ell+1)
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
        N = B.N_max - B.N_min(ell-B.R_max)
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


def inertial_waves(B, m, domain):
    print('Computing Inertial Wave Solutions')

    alpha_BC = 0
    save_plots = True

    if save_plots:
        def savefig(fn): plt.savefig(fn)
    else:
        def savefig(_): pass

    # Construct the state vector
    state_vector = StateVector(B, 'mlr', [('u',1),('p',0)], ntau=ntau, m_min=m, m_max=m)

    # Build up the linear system
    print('  Building matrices')
    Amat, Bmat = build_matrices(B, state_vector, m, alpha_BC=alpha_BC)

    # Compute eigenvalues
    print('  Solving eigenproblem for m = {}, size {}x{}'.format(m, np.shape(Amat)[0], np.shape(Amat)[1]))
    lam, v = eigsort(Amat.todense(), Bmat.todense())

    # Print the eigenvalues
    print('    m: {:2d},  largest real eigenvalue: {}'.format(m, lam[-1]))
    for l in lam:
        print("    m = {:2d}, evalues = {}".format(m, l))

    # Analytic eigenvalues are in real and in the interval [-2,2].
    # They are solutions to the eigenvalue problem:
    #   m * P_l^|m|(λ/2) = 2*(1 - λ**2/4) * d/dλ P_l^|m|(λ/2),
    dirtylam = np.copy(lam)
    bad = np.logical_or(np.abs(lam.imag) > 1e-12, np.abs(lam.real) > 2)
    lam[bad] = np.nan
    v = v[:,np.isfinite(lam)]
    lam = lam[np.isfinite(lam)]

    ntotal = np.shape(Amat)[0]
    print("  Number of good eigenvalues = {}/{}".format(len(lam), ntotal))

    plt.plot(np.real(dirtylam), np.imag(dirtylam), '.', markersize=3, color='tab:orange')
    plt.plot(np.real(lam), np.imag(lam), '.', markersize=3, color='tab:blue')
    plt.xlabel('Real(λ)')
    plt.ylabel('Imag(λ)')
    plt.title('Inertial Wave Eigenvalues, m = {}, Lmax = {}'.format(m, B.L_max))
    plt.xlim([-2.1,2.1])
    plt.ylim([-1,1])
    plt.grid(True)
    # filename = 'figures/inertial_wave_eigenvalues-m={}-Lmax={}'.format(m, B.L_max)
    # savefig(filename + '.eps')

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

    target_ids = [key for key in targets.keys() if key[2] == m]
    for index in target_ids:
        modestr = str(index[0]) + str(index[1]) + str(index[2])

        eval_target = targets[index]
        ind = np.argmin(abs(lam - eval_target))
        eval = lam[ind]
        evec = v[:, ind]

        print("  Plotting eigenvectors for Greenspan Mode {}, λ = {:5f}...".format(index,eval))
        # Unpack the eigenvector into our tensor fields
        u = ball.TensorField_3D(1, B, domain)
        p = ball.TensorField_3D(0, B, domain)
        state_vector.unpack(evec, [u, p])

        u, r, theta, phi = dealias(B, domain, u, L_factor=5/2, N_factor=5/2)
        ur, utheta, uphi = u['g'][0], u['g'][1], u['g'][2]

        ux = ur * np.sin(theta) * np.cos(phi) + utheta * np.cos(theta) * np.cos(phi) - uphi * np.sin(phi)
        uy = ur * np.sin(theta) * np.sin(phi) + utheta * np.cos(theta) * np.sin(phi) + uphi * np.cos(phi)
        uz = ur * np.cos(theta)               - utheta * np.sin(theta)

        plotmeridionalquiver(ux, uz, r, theta, phi)
        plt.title('Mode {}'.format(index))
        filename = 'figures/inertial_wave-mode={}-quiver.png'.format(modestr)
        savefig(filename)

        # plotmeridionalquiver(uy, uz, r, theta, phi)
        # plt.title('Meridional, Mode = {}, y-z'.format(index))

        continue

        # Dealias for plotting
        res = 256
        L_factor, N_factor = res // (B.L_max + 1), res // (B.N_max + 1)
        u, r, theta, phi = dealias(B, domain, u, L_factor=L_factor, N_factor=N_factor)
        ur, utheta, uphi = u['g'][0], u['g'][1], u['g'][2]

        filename = lambda field, sl: 'figures/inertial_wave-mode={}-field={}-slice={}.png'.format(modestr, field, sl)

        # Plot
        if m > 0:
            plotequatorialslice(ur, r, theta, phi)
            plt.title('Greenspan Mode {}, Equatorial Slice, $u_r$'.format(index))
            savefig(filename('ur', 'e'))

            plotequatorialslice(utheta, r, theta, phi)
            plt.title('Greenspan Mode {}, Equatorial Slice, $u_Θ$'.format(index))
            savefig(filename('utheta', 'e'))

            plotequatorialslice(uphi, r, theta, phi)
            plt.title('Greenspan Mode {}, Equatorial Slice, $u_ϕ$'.format(index))
            savefig(filename('uphi', 'e'))

        angle = 0.

        plotmeridionalslice(ur, r, theta, phi, angle=angle)
        plt.title('Greenspan Mode {}, Meridional Slice, $u_r$'.format(index))
        savefig(filename('ur', 'm'))

        plotmeridionalslice(utheta, r, theta, phi, angle=angle)
        plt.title('Greenspan Mode {}, Meridional Slice, $u_Θ$'.format(index))
        savefig(filename('utheta', 'm'))

    plt.show()
    return lam, v

def main():
    # Create the domain
    m = 1
    L_max = 15
    B, domain = build_ball(L_max=L_max, N_max=L_max)
    lam, v = inertial_waves(B, m, domain)


if __name__=='__main__':
    main()

