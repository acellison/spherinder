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
from eigtools import scipy_sparse_eigs, track_eigenpair, eigsort
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
        M_ell, L_ell = boussinesq.matrices(B,N,ell,Ekman,Prandtl,Rayleigh,alpha_BC,implicit_buoyancy=True,implicit_base_temp=True)
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


def build_unscaled_matrices(B, state_vector, m, alpha_BC=0):
    ell_range = range(m, B.L_max+1)

    time_start = time.time()
    M, L, E = build_matrices_ell(B, Ekman=1, Prandtl=1, Rayleigh=1, ell_range=ell_range, alpha_BC=alpha_BC)
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


def rayleigh_bisection(B, m, Lunscaled, Munscaled, Cor, lam, v, Ekman, Prandtl, Rayleigh, bracket_scale=1.1,
                       newton_tol=1e-9, bisect_tol=1e-3, max_newton_iters=12, max_bisect_iters=12):
    # Check current Rayleigh number
    Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)
    lam, v = track_eigenpair(Amat, Bmat, lam, v, tol=newton_tol, maxiter=max_newton_iters, verbose=False)

    print('  Bracketing critical Rayleigh number for Ekman = {:1.5e},  Eigenvalue = {:1.5e}...'.format(Ekman, lam), flush=True)
    Ra_min, Ra_max = Rayleigh, Rayleigh
    moveup = lam.real < 0
    if moveup:
        while lam.real <= 0:
            Rayleigh *= bracket_scale
            Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)
            lam, v = track_eigenpair(Amat, Bmat, lam, v, tol=newton_tol, maxiter=max_newton_iters, verbose=False)
            Ra_min = Ra_max
            Ra_max = Rayleigh
            print('    Rayleigh = {:1.5e},  Eigenvalue = {:1.5e}...'.format(Rayleigh, lam), flush=True)
    else:
        while lam.real >= 0:
            Rayleigh /= bracket_scale
            Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)
            lam, v = track_eigenpair(Amat, Bmat, lam, v, tol=newton_tol, maxiter=max_newton_iters, verbose=False)
            Ra_max = Ra_min
            Ra_min = Rayleigh
            print('    Rayleigh = {:1.5e},  Eigenvalue = {:1.5e}...'.format(Rayleigh, lam), flush=True)

    iter, done = 0, False
    while not done:
        # Build the matrices for the updated Rayleigh number
        Rayleigh = 0.5 * (Ra_min + Ra_max)
        Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)

        # Track the eigenpair as Rayleigh changed
        lam, v = track_eigenpair(Amat, Bmat, lam, v, tol=newton_tol, maxiter=max_newton_iters, verbose=False)

        # Update the bisection brackets
        if np.real(lam) > 0:
            Ra_max = Rayleigh
        else:
            Ra_min = Rayleigh

        iter += 1
        print('  Bisection Iteration: {:3d},  Ekman: {:1.5e},  Rayleigh number: {:1.5e},  λ = {: 1.9e}'.format(iter,
                                                                                                               Ekman,
                                                                                                               Rayleigh,
                                                                                                               lam),
              flush=True)
        done = np.abs(np.real(lam)) <= bisect_tol or iter > max_bisect_iters

    return lam, v, Rayleigh


def compute_critical_rayleigh(B, m, domain):
    print('Computing Critical Rayleigh Number', flush=True)
    print('  Boussinesq ball dimensions: m = {}, L_max = {}, N_max = {}'.format(m, B.L_max, B.N_max), flush=True)

    plotevec = True
    save_plots = True
    plot_resolution = 512
    plot_dpi = 900

    # Jones + Marti cases
    critical_pairs = {9: (1e9, 4.761e6, 4.428e2), 14: (1e10, 2.105e7, 9.849e2), 20: (1e11, 9.466e7, 2.124e3),
                      24: (3e11, 1.947e8, 3.073e3), 30: (1e12, 4.302e8, 4.638e3), 95: (1e15, 4.1742e10, 4.6827e4),
                      # 139: (1e16, 1.9281e11, 1.00675e5)}
                      139: (1e16, 1.92802e11, 1.004e5)}
    Ta, Ra, omega = critical_pairs[m]
    Ekman = 1/Ta**0.5
    Prandtl = 1
    Rayleigh = Ekman * Ra
    alpha_BC = 2

    # Ekman_range = Ekman*np.logspace(0,-2,21)
    Ekman_range = [Ekman]

    bisect_tol = 1e-2
    newton_tol = 1e-10
    max_newton_iters = 12
    max_bisect_iters = 24
    lamtarget = 1e3 - 1j*omega

    if save_plots:
        def savefig(fn): plt.savefig(fn + '.png', dpi=plot_dpi)
    else:
        def savefig(_): pass

    print('  Constructing the full system, m = {}, Ekman = {}, Rayleigh = {}...'.format(m, Ekman, Rayleigh), flush=True)
    ntau = lambda ell: 1 if ell == 0 else 4
    fields = [('u', 1), ('p', 0), ('T', 0)]
    state_vector = StateVector(B, 'mlr', fields, ntau=ntau, m_min=m, m_max=m)

    time_start = time.time()
    Lunscaled, Munscaled, Cor = build_unscaled_matrices(B, state_vector, m, alpha_BC)

    t1 = time.time()
    Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)
    t2 = time.time()
    print('    Non-dimensional scaling took {:g} sec'.format(t2-t1), flush=True)

    time_end = time.time()
    print('    Complete system construction took {:g} sec'.format(time_end-time_start), flush=True)

    print('  Solving sparse eigenproblem for m = {}, L_max = {}, N_max = {}, size {}x{}'.format(m, B.L_max, B.N_max, np.shape(Amat)[0], np.shape(Amat)[1]), flush=True)
    lam, v = scipy_sparse_eigs(Amat, Bmat, N=1, target=lamtarget, profile=True)
    lam = lam[0]

    Rayleigh_critical = []
    omega_critical = []

    """
    # Track the eigenpair to find the critical rayleigh number
    for i in range(len(Ekman_range)):
        Ekman = Ekman_range[i]

        lam, v, Rayleigh = rayleigh_bisection(B, m, Lunscaled, Munscaled, Cor, lam, v, Ekman, Prandtl, Rayleigh,
                                              bracket_scale=1.02, newton_tol=newton_tol, bisect_tol=bisect_tol,
                                              max_newton_iters=max_newton_iters, max_bisect_iters=max_bisect_iters)

        check = False
        if check:
            Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)
            lam2, v2 = scipy_sparse_eigs(Amat, Bmat, N=4, target=lamtarget, profile=True)
            print(lam2)
            lam2 = lam2[-1]
            error = np.abs(lam-lam2)
            if error > 1e-1:
                raise ValueError('Eigenvalue fell off the rails: mine: {:1.5e}, scipy: {:1.5e}, error: {:1.5e}'.format(lam, lam2, error))

        print('  Critical Rayleigh number for system [m = {}, Ekman = {:1.5e}] = {},  Reduced Rayleigh = {:1.5e},  ω = {:1.5e}'.format(m, Ekman, Rayleigh/Ekman, Rayleigh, -lam.imag), flush=True)
        Rayleigh_critical.append(Rayleigh/Ekman)
        omega_critical.append(lam.imag)

    print(Ekman_range)
    print(Rayleigh_critical)
    for i in range(len(Ekman_range)):
        print('Ekman: {:1.5e}, Rayleigh: {:1.5e}, omega: {:1.5e}'.format(Ekman_range[i], Rayleigh_critical[i], omega_critical[i]))
    """

    if plotevec:
        eval = lam
        evec = v

        print('  Plotting eigenvector for eigenvalue = {}'.format(eval), flush=True)
        # Unpack the eigenvector into our tensor fields
        u = ball.TensorField_3D(1, B, domain)
        p = ball.TensorField_3D(0, B, domain)
        T = ball.TensorField_3D(0, B, domain)
        state_vector.unpack(evec, [u, p, T])

        plt.figure()
        for ell in range(m, B.L_max+1):
            Tv = [np.abs(T['c'][ell][n, m]) for n in range(np.shape(T['c'][ell])[0])]
            plt.semilogy(Tv)
        plt.grid(True)
        plt.xlabel('Radial degree')

        plt.figure()
        Tv = []
        ell_range = range(m, B.L_max+1)
        for ell in ell_range:
            Tv.append(np.linalg.norm(T['c'][ell][:, m]))
        plt.semilogy(ell_range, Tv)
        plt.grid(True)
        plt.xlabel('Harmonic degree l')

        # Dealias for plotting
        L_factor, N_factor = max(plot_resolution // (B.L_max + 1), 1), max(plot_resolution // (B.N_max + 1), 1)

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
    import warnings
    warnings.simplefilter("ignore")

    # Create the domain
    mcritical = [9, 14, 20, 24, 30, 95, 139]
    m = mcritical[6]
    L_max, N_max = 401, 301
    B, domain = build_ball(L_max=L_max, N_max=N_max)
    Rayleigh, lam, v = compute_critical_rayleigh(B, m, domain)
    plt.show()


if __name__=='__main__':
    main()

