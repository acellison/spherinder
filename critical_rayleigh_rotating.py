from dedalus_sphere import ball_wrapper as ball
import dedalus.public as de
from dedalus.core.distributor import Distributor
import numpy as np
import scipy.sparse as sparse
from mpi4py import MPI
import matplotlib.pyplot as plt
import boussinesq
from state_vector import StateVector
from eigtools import eigsort, scipy_sparse_eigs
from plot_tools import plotequatorialslice, plotmeridionalslice, dealias, sph2cart


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


def build_matrices_ell(B, Ekman, Prandtl, Rayleigh, ell_range, alpha_BC):
    M, L, E = [], [], []
    for ell in ell_range:
        N = B.N_max - B.N_min(ell-B.R_max)
        M_ell, L_ell = boussinesq.matrices(B,N,ell,Ekman,Prandtl,Rayleigh,alpha_BC,implicit_buoyancy=True,implicit_base_temp=True, divtau=False)
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


def build_matrices(B, state_vector, m, Ekman, Prandtl, Rayleigh, alpha_BC):
    ell_range = range(m, B.L_max+1)
    M, L, E = build_matrices_ell(B, Ekman, Prandtl, Rayleigh, ell_range=ell_range, alpha_BC=alpha_BC)

    # Outer blocks of m, then each ell gets all its r
    Mm = sparse.block_diag([M_ell for M_ell in M])
    Lm = sparse.block_diag([L_ell for L_ell in L])
    Em = sparse.block_diag([E_ell for E_ell in E])

    Cm = boussinesq.create_coriolis_matrix(B, state_vector, m_min=m, m_max=m)
    Cm = Em.dot(Cm)

    Amat = Lm + Cm
    Bmat = -Mm

    return Amat, Bmat


def compute_critical_rayleigh(B, m, domain):
    print('Computing Critical Rayleigh Number', flush=True)
    print('  Boussinesq ball dimensions: m = {}, L_max = {}, N_max = {}'.format(m, B.L_max, B.N_max), flush=True)

    solvedense = False
    plotevec = True
    save_plots = True

    # Jones + Marti cases
    critical_pairs = {9: (1e9, 4.761e6, 4.428e2), 14: (1e10, 2.105e7, 9.849e2), 20: (1e11, 9.466e7, 2.124e3),
                      24: (3e11, 1.947e8, 3.073e3), 30: (1e12, 4.302e8, 4.638e3), 95: (1e15, 4.1742e10, 4.6827e4)}
    Ta, Ra, target = critical_pairs[m]
    Ekman = 1/Ta**0.5
    Prandtl = 1
    Rayleigh = Ekman * Ra
    alpha_BC = 0

    if save_plots:
        def savefig(fn): plt.savefig(fn + '.png', dpi=900)
    else:
        def savefig(_): pass

    print('  Constructing the full system, m = {}, Ekman = {}, Rayleigh = {}...'.format(m, Ekman, Rayleigh), flush=True)
    ntau = lambda ell: 1 if ell == 0 else 4
    fields = [('u', 1), ('p', 0), ('T', 0)]
    state_vector = StateVector(B, 'mlr', fields, ntau=ntau, m_min=m, m_max=m)
    Amat, Bmat = build_matrices(B, state_vector, m, Ekman, Prandtl, Rayleigh, alpha_BC=alpha_BC)

    if solvedense:
        print('  Densifying matrices', flush=True)
        Amat, Bmat = Amat.todense(), Bmat.todense()

        # Compute eigenvalues
        print('  Solving dense eigenproblem for m = {}, L_max = {}, N_max = {}, size {}x{}'.format(m, B.L_max, B.N_max, np.shape(Amat)[0], np.shape(Amat)[1]), flush=True)
        lam, v = eigsort(Amat, Bmat, profile=True)

    else:
        print('  Solving sparse eigenproblem for m = {}, L_max = {}, N_max = {}, size {}x{}'.format(m, B.L_max, B.N_max, np.shape(Amat)[0], np.shape(Amat)[1]), flush=True)
        lam, v = scipy_sparse_eigs(Amat, Bmat, N=1, target=-1j*target, profile=True)

    # Plot
    print('  Critical eigenvalue: {}'.format(lam[-1]), flush=True)
    plt.plot(np.real(lam), np.imag(lam), '.', markersize=3, color='tab:blue')
    plt.xlim([-15e3,300])
    plt.ylim([-1e5,1e5])
    plt.xlabel('Real(λ)')
    plt.ylabel('Imag(λ)')
    plt.title('Boussinesq Eigenvalues, m = {}, Lmax = {}'.format(m, B.L_max))
    plt.grid(True)
    filename = 'figures/boussinesq_eigenvalues-m={}-Lmax={}-Nmax={}-Ekman={:1.4e}-Rayleigh={:1.4e}'.format(m, B.L_max, B.N_max, Ekman, Rayleigh)
    savefig(filename)

    if plotevec:
        eval = lam[-1]
        evec = v[:, -1]

        print('  Plotting eigenvector for eigenvalue = {}'.format(eval), flush=True)
        # Unpack the eigenvector into our tensor fields
        u = ball.TensorField_3D(1, B, domain)
        p = ball.TensorField_3D(0, B, domain)
        T = ball.TensorField_3D(0, B, domain)
        state_vector.unpack(evec, [u, p, T])

        # Dealias for plotting
        res = 512
        L_factor, N_factor = res // (B.L_max + 1), res // (B.N_max + 1)

        filename = lambda field, sl: 'figures/boussinesq-m={}-Lmax={}-Nmax={}-field={}-slice={}-Ekman={:1.4e}-Rayleigh={:1.4e}'.format(m, B.L_max, B.N_max, field, sl, Ekman, Rayleigh)

        # Plot
        angle = 0.

        u, r, theta, phi = dealias(B, domain, u, L_factor=L_factor, N_factor=N_factor)
        ur, utheta, uphi = u['g'][0], u['g'][1], u['g'][2]
        ux, uy, uz = sph2cart(u, theta, phi)

        plotequatorialslice(ur, r, theta, phi)
        plt.title('Equatorial Slice, $u_r$')
        savefig(filename('ur', 'e'))

        plotequatorialslice(utheta, r, theta, phi)
        plt.title('Equatorial Slice, $u_Θ$')
        savefig(filename('utheta', 'e'))

        plotequatorialslice(uphi, r, theta, phi)
        plt.title('Equatorial Slice, $u_ϕ$')
        savefig(filename('uphi', 'e'))

        plotequatorialslice(ux, r, theta, phi)
        plt.title('Equatorial Slice, $u_x$')
        savefig(filename('ux', 'e'))

        plotequatorialslice(ux, r, theta, phi)
        plt.title('Equatorial Slice, $u_y$')
        savefig(filename('uy', 'e'))

        plotmeridionalslice(ur, r, theta, phi, angle=angle)
        plt.title('Meridional Slice, $u_r$')
        savefig(filename('ur', 'm'))

        plotmeridionalslice(utheta, r, theta, phi, angle=angle)
        plt.title('Meridional Slice, $u_Θ$')
        savefig(filename('utheta', 'm'))

        plotmeridionalslice(uphi, r, theta, phi, angle=angle)
        plt.title('Meridional Slice, $u_ϕ$')
        savefig(filename('uphi', 'm'))

        plotmeridionalslice(ux, r, theta, phi, angle=angle)
        plt.title('Meridional Slice, $u_x$')
        savefig(filename('ux', 'm'))

        plotmeridionalslice(uz, r, theta, phi, angle=angle)
        plt.title('Meridional Slice, $u_z$')
        savefig(filename('uz', 'm'))

    return Rayleigh, lam, v


def main():
    # Create the domain
    mcritical = [9, 14, 20, 24, 30, 95]
    m = mcritical[5]
    L_max, N_max = 255, 255
    B, domain = build_ball(L_max=L_max, N_max=N_max)
    Rayleigh, lam, v = compute_critical_rayleigh(B, m, domain)
    plt.show()


if __name__=='__main__':
    main()

