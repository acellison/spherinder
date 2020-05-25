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
from eigtools import scipy_sparse_eigs, track_eigenpair, discard_spurious_eigenvalues
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


def plot_spectral_decay(u, p, T, B, m, Ekman, Rayleigh, save_plots):
    filename = lambda field: 'figures/boussinesq-coeffs-m={}-Lmax={}-Nmax={}-Ekman={:1.4e}-Rayleigh={:1.9e}-kind={}'\
            .format(m, B.L_max, B.N_max, Ekman, Rayleigh,  field)

    if save_plots:
        def savefig(fn): plt.savefig(fn + '.png')
    else:
        def savefig(_): pass

    plt.figure()
    Tradial = []
    for ell in range(m, B.L_max + 1):
        Tradial.append(np.array([np.abs(T['c'][ell][n, m]) for n in range(np.shape(T['c'][ell])[0])]))
    factor = 1 / np.max(Tradial[0])
    for i in range(len(Tradial)):
        Tradial[i] *= factor
    for i in range(len(Tradial)):
        plt.semilogy(Tradial[i])
    plt.grid(True)
    plt.title('Magnitude of radial coefficients for each harmonic degree')
    plt.xlabel('Radial degree')
    savefig(filename('radial'))

    def plot_radial_norm(field, rank=0, a=0):
        Tell = []
        ell_range = range(m, B.L_max + 1)

        for ell in ell_range:
            nsize = np.shape(field['c'][ell])[0]
            n1 = a * (nsize // 3 ** rank)
            n2 = (a + 1) * (nsize // 3 ** rank)
            Tell.append(np.linalg.norm(field['c'][ell][n1:n2, m]))
        Tell = np.array(Tell)
        Tell /= max(Tell)
        plt.semilogy(ell_range, Tell)
        plt.grid(True)
        plt.title('Norm of radial coefficients for each harmonic degree')
        plt.xlabel('Harmonic degree l')

    # plt.figure()
    # plot_radial_norm(u, rank=1, a=0)
    # plot_radial_norm(u, rank=1, a=1)
    # plot_radial_norm(u, rank=1, a=2)
    # savefig(filename('ell'))

    plt.figure()
    plot_radial_norm(T)
    savefig(filename('ell'))

    # plt.figure()
    # plot_radial_norm(p)
    # savefig(filename('ell'))


def plot_fields(u, p, T, B, m, domain, Ekman, Rayleigh, save_plots, plot_resolution, plot_dpi):
    if save_plots:
        def savefig(fn): plt.savefig(fn + '.png', dpi=plot_dpi)
    else:
        def savefig(_): pass

    # Dealias for plotting
    L_factor, N_factor = max(plot_resolution // (B.L_max + 1), 1), max(plot_resolution // (B.N_max + 1), 1)

    filename = lambda field, sl: 'figures/stretch/boussinesq-m={}-Lmax={}-Nmax={}-field={}-slice={}-Ekman={:1.6e}-Rayleigh={:1.6e}'.format(
        m, B.L_max, B.N_max, field, sl, Ekman, Rayleigh)

    # Plot settings
    angle = 0.
    stretch = True

    # Compute the vorticity
    om = ball.TensorField_3D(1, B, domain)
    om.layout = 'c'
    for ell in range(m, B.L_max + 1):
        B.curl(ell, 1, u['c'][ell], om['c'][ell])

    # Dealias the fields
    T, _, _, _ = dealias(B, domain, T, L_factor=L_factor, N_factor=N_factor)
    T = T['g'][0]
    u, r, theta, phi = dealias(B, domain, u, L_factor=L_factor, N_factor=N_factor)
    ur, utheta, uphi = u['g'][0], u['g'][1], u['g'][2]

    # Dealias the vorticity field
    om, _, _, _ = dealias(B, domain, om, L_factor=L_factor, N_factor=N_factor)
    omz = om['g'][0] * np.cos(theta) - om['g'][1] * np.sin(theta)

    # plotequatorialslice(uphi, r, theta, phi)
    # plt.title('Equatorial Slice, $u_ϕ$')
    # savefig(filename('uphi', 'e'))

    # plotmeridionalslice(ur, r, theta, phi, angle=angle, stretch=stretch)
    # plt.title('Meridional Slice, $u_r$')
    # savefig(filename('ur', 'm'))

    plotmeridionalslice(uphi, r, theta, phi, angle=angle, stretch=stretch)
    plt.title('Meridional Slice, $u_ϕ$')
    savefig(filename('uphi', 'm'))

    plotmeridionalslice(omz, r, theta, phi, angle=angle, stretch=stretch)
    plt.title('Meridional Slice, $ω_z$')
    savefig(filename('omz', 'm'))

    plotmeridionalslice(T, r, theta, phi, angle=angle, stretch=stretch)
    plt.title('Meridional Slice, $T$')
    savefig(filename('T', 'm'))

    # Normalized kinetic energy
    ke = np.log10(0.5 * (u['g'][0] ** 2 + u['g'][1] ** 2 + u['g'][2] ** 2))
    ke -= np.max(ke)
    truncate_level = -8
    ketrunc = np.where(ke < truncate_level, np.nan, ke)

    # plotequatorialslice(ketrunc, r, theta, phi, cmap='RdBu_r')
    # plt.title('Equatorial Slice, $log_{10}$(Kinetic Energy), Truncated')
    # savefig(filename('ke', 'e'))

    plotmeridionalslice(ketrunc, r, theta, phi, angle=angle, stretch=stretch, cmap='RdBu_r')
    plt.title('Meridional Slice, $log_{10}$(Kinetic Energy), Truncated')
    savefig(filename('ke', 'm'))


def rayleigh_bisection(B, m, state_vector, Lunscaled, Munscaled, Cor, lam, v, Ekman, Prandtl, Rayleigh, bracket_scale=1.1,
                       newton_tol=1e-9, bisect_tol=1e-3, max_newton_iters=12, max_bisect_iters=12, verbose=False):
    if max_bisect_iters == 0:
        return lam, v, Rayleigh

    utilize_symmetry = False

    if utilize_symmetry:
        matsolver = 'LSQR_solve'
        extract_symmetry = lambda a: extract_symmetric_parts(a.tocsc(), B, m, state_vector)
    else:
        matsolver = 'SuperluNaturalSpsolve'
        extract_symmetry = lambda a: a

    eig = lambda Amat, Bmat, lam, v: track_eigenpair(Amat, Bmat, lam, v, matsolver=matsolver, tol=newton_tol,
                                                     maxiter=max_newton_iters, verbose=verbose)

    # Check current Rayleigh number
    print('  Building matrices to bracket the critical Rayleigh number...', flush=True)
    Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)

    Amat = extract_symmetry(Amat)
    Bmat = extract_symmetry(Bmat)
    if utilize_symmetry:
        v = extract_symmetric_parts(np.reshape(v, (1, state_vector.dof)), B, m, state_vector).T

    print('  Computing initial eigenpair...', flush=True)
    lam, v = eig(Amat, Bmat, lam, v)

    print('  Bracketing critical Rayleigh number for Ekman = {:1.5e},  Eigenvalue = {:1.5e}...'.format(Ekman, lam), flush=True)
    Ra_min, Ra_max = Rayleigh, Rayleigh
    moveup = lam.real < 0
    if moveup:
        while lam.real <= 0:
            Rayleigh *= bracket_scale
            Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)
            Amat = extract_symmetry(Amat)
            Bmat = extract_symmetry(Bmat)

            lam, v = eig(Amat, Bmat, lam, v)
            Ra_min = Ra_max
            Ra_max = Rayleigh
            print('    Rayleigh = {:1.5e},  Eigenvalue = {:1.5e}...'.format(Rayleigh, lam), flush=True)
    else:
        while lam.real >= 0:
            Rayleigh /= bracket_scale
            Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)
            Amat = extract_symmetry(Amat)
            Bmat = extract_symmetry(Bmat)

            lam, v = eig(Amat, Bmat, lam, v)
            Ra_max = Ra_min
            Ra_min = Rayleigh
            print('    Rayleigh = {:1.5e},  Eigenvalue = {:1.5e}...'.format(Rayleigh, lam), flush=True)

    print('  Bisection brackets for critical Rayleigh number: {}'.format((Ra_min,Ra_max)), flush=True)
    iter, done = 0, False
    while not done:
        iter += 1
        Rayleigh = 0.5 * (Ra_min + Ra_max)

        # Build the matrices for the updated Rayleigh number
        print('  Bisection Iteration: {:3d},  Building matrices for Rayleigh = {:1.9e}'.format(iter, Rayleigh), flush=True)
        Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)
        Amat = extract_symmetry(Amat)
        Bmat = extract_symmetry(Bmat)

        # Track the eigenpair as Rayleigh changed
        print('    Tracking eigenpair...', flush=True)
        lam, v = eig(Amat, Bmat, lam, v)

        # Update the bisection brackets
        if np.real(lam) > 0:
            Ra_max = Rayleigh
        else:
            Ra_min = Rayleigh

        print('    Ekman: {:1.5e},  Rayleigh number: {:1.9e},  λ = {: 1.9e}'.format(iter, Ekman, Rayleigh, lam), flush=True)
        done = np.abs(np.real(lam)) <= bisect_tol or iter >= max_bisect_iters

    if utilize_symmetry:
        v = restore_asymmetric_parts(v, B, m, state_vector)
    v = np.reshape(np.asarray(v.ravel()), np.prod(np.shape(v)))
    return lam, v, Rayleigh


def get_symmetric_indices(B, m, state_vector):
    columns = []
    for ell in range(m, B.L_max+1):
        first = state_vector.index('u', ell, 0, m)
        usize = state_vector.index('p', ell, 0, m) - first
        taustart = state_vector.index('tau', ell, 0, m)

        if (ell - m) % 2 == 0:
            # Even ell - take everything but the u0 component
            index1 = first
            index2 = first + usize//3
            columns += list(range(index1, index2))

            index1 = first + 2*usize//3
            index2 = taustart
            columns += list(range(index1, index2))

            # Tau error in u-, u+ and T components
            if ell > 0:
                columns += [index2, index2+2, index2+3]
            else:
                columns += [index2]
        else:
            # Odd ell - take only the u0 component
            index1 = first +   usize//3
            index2 = first + 2*usize//3
            columns += list(range(index1, index2))

            # Tau error in the u0 component
            if ell > 0:
                columns += [taustart+1]
    return columns


def extract_symmetric_parts(A, B, m, state_vector):
    columns = get_symmetric_indices(B, m, state_vector)
    return A[:, columns]


def restore_asymmetric_parts(v, B, m, state_vector):
    indices = get_symmetric_indices(B, m, state_vector)
    vout = np.zeros((state_vector.dof, 1), dtype=v.dtype)
    vout[indices, 0] = v
    return vout


def compute_critical_rayleigh(B, m, domain, nev=10, evalue_only=False):
    print('Computing Critical Rayleigh Number', flush=True)
    print('  Boussinesq ball dimensions: m = {}, L_max = {}, N_max = {}'.format(m, B.L_max, B.N_max), flush=True)

    plot_evec = True
    plot_coeff_decay = False
    save_plots = True
    plot_resolution = 256
    plot_dpi = 600

    # Jones + Marti cases
    critical_pairs = {9: (1e9, 4.761e6, 4.428e2), 14: (1e10, 2.105e7, 9.849e2), 20: (1e11, 9.466e7, 2.124e3),
                      24: (3e11, 1.947e8, 3.073e3), 30: (1e12, 4.302e8, 4.638e3), 95: (1e15, 4.1742e10, 4.6827e4),
                      139: (1e16, 1.92691e11, 1e5)}
    Ta, Ra, omega = critical_pairs[m]
    Ekman = 1/Ta**0.5
    Prandtl = 1
    Rayleigh = Ekman * Ra
    alpha_BC = 2

    # Ekman_range = Ekman*np.logspace(0,-2,21)
    Ekman_range = [Ekman]

    # lamtarget = 1e3 - 1j*omega
    lamtarget = -1j*omega
    verbose = True
    bisect_tol = 1e-4
    newton_tol = 1e-10
    max_newton_iters = 5
    max_bisect_iters = 0
    check_critical_rayleigh = True

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
    lam, v = scipy_sparse_eigs(Amat, Bmat, N=nev, target=lamtarget, profile=True)

    print('  Computed Eigenvalues:')
    for i in range(len(lam)):
        print('    evalue = {: 1.9e}'.format(lam[i]))
    print('  Eigenvalue with maximum real part = {: 1.9e}'.format(lam[-1]), flush=True)

    if evalue_only:
        return lam

    lam = lam[-1]
    v = v[:,-1]
    Rayleigh_critical = []
    omega_critical = []

    # Track the eigenpair to find the critical rayleigh number
    for i in range(len(Ekman_range)):
        Ekman = Ekman_range[i]

        lam, v, Rayleigh = rayleigh_bisection(B, m, state_vector, Lunscaled, Munscaled, Cor, lam, v, Ekman, Prandtl, Rayleigh,
                                              bracket_scale=1.01, newton_tol=newton_tol, bisect_tol=bisect_tol,
                                              max_newton_iters=max_newton_iters, max_bisect_iters=max_bisect_iters,
                                              verbose=verbose)

        if check_critical_rayleigh and max_bisect_iters > 0:
            Amat, Bmat = build_matrices(B, m, Lunscaled, Munscaled, Cor, Ekman, Prandtl, Rayleigh)
            lam2, v2 = scipy_sparse_eigs(Amat, Bmat, N=nev, target=lamtarget, profile=True)
            lam2 = lam2[-1]
            error = np.abs(lam-lam2)
            print('  Error in Newton eigenvalue vs. scipy solve: {}'.format(error))
            if error > 1e-1:
                raise ValueError('Eigenvalue fell off the rails: mine: {:1.5e}, scipy: {:1.5e}, error: {:1.5e}'.format(lam, lam2, error))

        print('  Critical Rayleigh number for system [m = {}, Ekman = {:1.5e}] = {},  Reduced Rayleigh = {:1.9e},  ω = {:1.5e}'.format(m, Ekman, Rayleigh/Ekman, Rayleigh, -lam.imag), flush=True)
        Rayleigh_critical.append(Rayleigh/Ekman)
        omega_critical.append(lam.imag)

    for i in range(len(Ekman_range)):
        print('Ekman: {:1.5e}, Rayleigh: {:1.9e}, omega: {:1.5e}'.format(Ekman_range[i], Rayleigh_critical[i], omega_critical[i]))

    u, p, T = None, None, None
    if plot_coeff_decay or plot_evec:
        eval = lam
        evec = v

        print('  Plotting eigenvector for eigenvalue = {}'.format(eval), flush=True)
        # Unpack the eigenvector into our tensor fields
        u = ball.TensorField_3D(1, B, domain)
        p = ball.TensorField_3D(0, B, domain)
        T = ball.TensorField_3D(0, B, domain)
        state_vector.unpack(evec, [u, p, T])

    if plot_coeff_decay:
        plot_spectral_decay(u, p, T, B, m, Ekman, Rayleigh, save_plots)

    if plot_evec:
        plot_fields(u, p, T, B, m, domain, Ekman, Rayleigh, save_plots, plot_resolution, plot_dpi)


def investigate_spuriosity():
    m = 139
    cutoff = 1e6
    nev = 10

    # resolutions = [(15,15), (23,23), (31,31), (47,47), (63,63)]
    resolutions = [(351,301), (401,301), (401,351), (451,351), (451,401), (501,401), (501,451)]
    evalues = []
    for i in range(len(resolutions)):
        L_max, N_max = resolutions[i][0], resolutions[i][1]
        B, domain = build_ball(L_max=L_max, N_max=N_max)
        lam = compute_critical_rayleigh(B, m, domain, nev=nev, evalue_only=True)
        evalues.append(lam)

    for i in range(len(resolutions)-1):
        lores = evalues[i]
        hires = evalues[-1]
        good = discard_spurious_eigenvalues(lores, hires, cutoff, plot=True)
        print("Number of good eigenvalues for resolution {}: {}/{}".format(resolutions[i], len(good), len(lores)))
        print(good)
    plt.show()


def main():
    import warnings
    warnings.simplefilter("ignore")

    resolution = {9:(31,31), 14:(47,47), 20:(63,63), 24:(79,79),
                  30:(121,83), 95:(295,201), 139:(351,301)}

    mvals = [9, 14, 30, 95]
    for m in mvals:
        L_max, N_max = resolution[m]
        B, domain = build_ball(L_max=L_max, N_max=N_max)
        compute_critical_rayleigh(B, m, domain)

    plt.show()


if __name__=='__main__':
    main()
    # investigate_spuriosity()
