from dedalus_sphere import ball_wrapper as ball
import dedalus.public as de
from dedalus.core.distributor import Distributor
import numpy as np
from scipy.linalg import eig
from mpi4py import MPI
import time
import boussinesq
from eigtools import track_eigenpair


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

    return B


def build_matrices(B, Ekman, Prandtl, Rayleigh, ell_range, alpha_BC=0):
    # build matrices
    M,L = [],[]
    for ell in ell_range:
        N = B.N_max - B.N_min(ell-B.R_max)
        M_ell,L_ell = boussinesq.matrices(B,N,ell,Ekman,Prandtl,Rayleigh,alpha_BC,implicit_buoyancy=True,implicit_base_temp=True)
        M.append(M_ell.astype(np.complex128))
        L.append(L_ell.astype(np.complex128))

    return M, L


def compute_eigenvalues(M, L, ell_range=None):
    cutoff = np.inf

    timeit = False
    if timeit:
        print("Starting eigenvalue computation...")
        evals_start = time.time()

    evals, evecs = [], []
    for ell_local in range(len(ell_range)):
        L_local = L[ell_local].todense()
        M_local = M[ell_local].todense()
        vals, vecs = eig(L_local,b=-M_local)
        bad = (np.abs(vals) > cutoff)
        vals[bad] = np.nan
        vecs = vecs[:,np.isfinite(vals)]
        vals = vals[np.isfinite(vals)]

        i = np.argsort(vals.real)
        vals, vecs = vals[i], vecs.transpose()[i]

        evals.append(vals)
        evecs.append(vecs)

    if timeit:
        evals_end = time.time()
        print("Eigenvalues took {:g} sec".format(evals_end-evals_start))

    return evals, evecs


def compute_critical_rayleigh(B, ell):
    print('Computing Critical Rayleigh Number for ell = {}...'.format(ell))

    Ekman = 1
    Prandtl = 1
    rayleigh_brackets = [ (None, None),
        (3.0e3,  3.1e3),  (5.2e3,  5.4e3),  (8.7e3,  8.8e3),  (1.3e4,  1.4e4),  (2.12e4, 2.13e4),
        (3.08e4, 3.09e4), (4.33e4, 4.34e4), (5.91e4, 5.92e4), (7.88e4, 7.89e4), (1.02e5, 1.03e5),
        (1.31e5, 1.32e5), (1.66e5, 1.67e5), (2.07e5, 2.08e5), (2.54e5, 2.55e5), (3.09e5, 3.10e5)]

    bisect_tol = 1e-6
    newton_tol = 1e-15
    max_bisect_iters = 100

    Ra_min, Ra_max = rayleigh_brackets[ell]
    ell_range = [ell]
    alpha_BC = 2

    # Build up the linear system
    Rayleigh = Ra_max
    M, L = build_matrices(B, Ekman, Prandtl, Rayleigh, ell_range=ell_range, alpha_BC=alpha_BC)

    # Compute complete set of eigenvalues
    evals1, evecs1 = compute_eigenvalues(M, L, ell_range=ell_range)

    # Grab the eigenvalue with maximum real part to start
    lam, v = evals1[0][-1], evecs1[0][-1]
    if np.real(lam) <= 0:
        raise ValueError('Could not bracket the critical Rayleigh number - increase Ra_max')

    # Track the eigenpair to find the critical rayleigh number
    iter = 0
    while np.abs(np.real(lam)) > bisect_tol and iter < max_bisect_iters:
        Rayleigh = 0.5 * (Ra_min + Ra_max)
        M, L = build_matrices(B, Ekman, Prandtl, Rayleigh, ell_range=ell_range, alpha_BC=alpha_BC)
        M, L = M[0], L[0]

        lam, v = track_eigenpair(L, -M, lam, v, tol=newton_tol)

        if np.real(lam) > 0:
            Ra_max = Rayleigh
        else:
            Ra_min = Rayleigh

        iter += 1
        print('  ell = {:2d},  iteration: {:3d},  Rayleigh number: {:1.5e},  corresponding eigenvalue: {: 1.9e}'.format(ell, iter, Rayleigh, np.real(lam)))

    print('Critical Rayleigh Number for ell = {}: {:1.5e}'.format(ell, Rayleigh))

    return Rayleigh, lam, v


def main():
    # Create the domain
    N = 31
    B = build_ball(L_max=N, N_max=N)

    # Compute the critical Rayleigh numbers for all the Chandra cases
    ell_min = 1
    ell_max = 15
    ell_range = list(range(ell_min, ell_max+1))
    Rayleighs = {}
    for ell in ell_range:
        Ra, _, _ = compute_critical_rayleigh(B, ell=ell)
        Rayleighs[ell] = Ra

    # Print resulting critical Rayleigh numbers
    print('')
    for ell, Rayleigh in Rayleighs.items():
        print('Critical Rayleigh Number, ell = {:2d}:  {:1.5e}'.format(ell, Rayleigh))


if __name__=='__main__':
    main()

