from dedalus_sphere import ball_wrapper as ball
import dedalus.public as de
from dedalus.core.distributor import Distributor
from mpi4py import MPI
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
from boussinesq import create_coriolis_matrix
from state_vector import StateVector

# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Resolution
L_max = 31
N_max = 31
R_max = 3

L_dealias = 3/2
N_dealias = 3/2
N_r = N_max

# Make domain
mesh=[1]
phi_basis = de.Fourier('phi',2*(L_max+1), interval=(0,2*np.pi),dealias=L_dealias)
theta_basis = de.Fourier('theta', L_max+1, interval=(0,np.pi),dealias=L_dealias)
r_basis = de.Fourier('r', N_max+1, interval=(0,1),dealias=N_dealias)
domain = de.Domain([phi_basis,theta_basis,r_basis], grid_dtype=np.float64, mesh=mesh)

domain.global_coeff_shape = np.array([L_max+1,L_max+1,N_max+1])
domain.distributor = Distributor(domain,comm,mesh)

mesh = domain.distributor.mesh
phi_layout   = domain.distributor.layouts[3]
th_m_layout  = domain.distributor.layouts[2]
ell_r_layout = domain.distributor.layouts[1]
r_ell_layout = domain.distributor.layouts[1]

m_start   = th_m_layout.slices(scales=1)[0].start
m_end     = th_m_layout.slices(scales=1)[0].stop-1
m_size = m_end - m_start + 1
ell_start = r_ell_layout.slices(scales=1)[1].start
ell_end   = r_ell_layout.slices(scales=1)[1].stop-1

# set up ball
N_theta = int((L_max+1)*L_dealias)
N_r     = int((N_r+1)*N_dealias)
B = ball.Ball(N_max,L_max,N_theta=N_theta,N_r=N_r,R_max=R_max,ell_min=ell_start,ell_max=ell_end,m_min=m_start,m_max=m_end,a=0.)
theta_global = B.grid(0)
r_global = B.grid(1)
z, R = r_global*np.cos(theta_global), r_global*np.sin(theta_global) # global

grid_slices = phi_layout.slices(domain.dealias)
phi = domain.grid(0,scales=domain.dealias)[grid_slices[0],:,:]
theta = B.grid(1,dimensions=3)[:,grid_slices[1],:] # local
r = B.grid(2,dimensions=3)[:,:,grid_slices[2]] # local


def print_errors(coef, grid, rank=0, plot=False, tol=1e-16):
    if plot == False or plot is None:
        plot = -1
    # Compare in coefficient space
    for i in range(len(grid['c'])):
        rawerror = coef['c'][i]-grid['c'][i]
        error = abs(rawerror)
        maxerror = np.max(error)
        if maxerror > tol:
            if rank==1:
                u_len = coef['c'][i].shape[0]
                errorm = np.sum(error[:u_len//3, :])
                error0 = np.sum(error[u_len//3:2*u_len//3, :])
                errorp = np.sum(error[2*u_len//3:, :])
                print('Coef space: ell = {:2d},   max error: {:1.5e},   errorm: {:1.5e},   error0: {:1.5e},   errorp: {:1.5e}'.format(i,maxerror,errorm, error0, errorp))
                if plot >= 0:
                    which = plot
                    plt.plot(grid['c'][i][which*u_len//3:(which+1)*u_len//3, :].flatten()-coef['c'][i][which*u_len//3:(which+1)*u_len//3, :].flatten())
            else:
                l1error = np.sum(error)
                print('Coef space: ell = {:2d},   max error: {:1.5e},  L1 error: {:1.5e}'.format(i,maxerror,l1error))
                if plot >= 0:
                    plt.plot(grid['c'][i].flatten()-coef['c'][i].flatten())

    if plot is not None: plt.show()
    # Compare in grid space
    for i in range(len(grid['g'])):
        error = coef['g'][i].flatten()-grid['g'][i].flatten()
        ind = (i, *np.unravel_index(np.argmax(error, axis=None), coef['g'][i].shape))
        maxerror = np.max(abs(error))
        l1error = np.sum(abs(error))
        if maxerror > tol:
            print('Grid space: ind: {}, maxerror: {}, L1error: {}'.format(ind, maxerror, l1error))


def test_coriolis(layout, truncate=False):
    print('testing ez cross u with matrix multiply with layout {}...'.format(layout))

    # initial condition
    random_initial_condition = True
    uinit = ball.TensorField_3D(1, B, domain)
    ugrid = ball.TensorField_3D(1, B, domain)
    ucoef = ball.TensorField_3D(1, B, domain)

    if random_initial_condition:
        np.random.seed(0)
        uinit.layout = 'c'
        for ell in range(len(uinit['c'])):
            rows = np.shape(uinit['c'][ell])[0]
            cols = ell+1
            uinit['c'][ell][:rows,:cols] += np.random.random((rows,cols))
        uinit['g']
        uinit['c']
    else:
        uinit.layout = 'g'
        uinit['g'][0] = r*(np.cos(theta)**2 - r**2*np.cos(phi)*np.cos(theta)**3*np.sin(theta) + np.cos(phi)*(-1 + r*np.cos(theta))*np.sin(phi)*np.sin(theta)**2 +
             - r*np.sin(phi)**3*np.sin(theta)**3 + r**7*np.cos(phi)**8*np.cos(theta)*np.sin(theta)**8)
        uinit['g'][1] = -  -(r*(r**7*np.cos(phi)**8*np.sin(theta)**9 + np.cos(phi)*np.cos(theta)*(r**2*np.cos(theta)**3 + (1 - r*np.cos(theta))*np.sin(phi)*np.sin(theta)) +
             - np.cos(theta)*np.sin(theta)*(1 - r*np.sin(phi)**3*np.sin(theta))))
        uinit['g'][2] = -  r*(r**2*np.cos(theta)**3*np.sin(phi) + r*np.cos(phi)**2*np.cos(theta)*np.sin(theta) + np.sin(phi)**2*np.sin(theta)*(1 + r*np.cos(phi)*np.sin(theta)))
        uinit['c']

    # Compute er cos(theta) cross u in grid space
    ez = np.array([np.cos(theta),-np.sin(theta),0*np.cos(theta)])
    ugrid['g'] = B.cross_grid(ez, uinit['g'])
    ugrid['c']

    # Construct the coriolis matrix
    print('Constructing matrix...')
    if layout != 'mlr':
        truncate = False  # Not implemented yet
    state_vector = StateVector(B, layout, ('u',1), ntau=lambda _:0, truncate=truncate)
    Cor = create_coriolis_matrix(B, state_vector)

    print('Evaluating matrix-vector product...')

    # Flatten initial condition
    uflat = np.zeros(np.shape(Cor)[0], dtype=uinit['c'][0].dtype)
    if layout == 'mlr':
        offset = 0
        for m in range(L_max+1):
            ell_start = m if truncate else 0
            for ell in range(ell_start, L_max+1):
                ne = np.shape(uinit['c'][ell])[0]
                uflat[offset:offset+ne] = uinit['c'][ell][:,m]
                offset += ne
    else:
        offset = 0
        for ell in range(len(uinit['c'])):
            ne = np.prod(np.shape(uinit['c'][ell]))
            if layout == 'lrm':
                uflat[offset:offset+ne] = uinit['c'][ell].ravel()
            elif layout == 'lmr':
                uflat[offset:offset + ne] = uinit['c'][ell].T.ravel()
            offset += ne

    # Apply Coriolis
    uflat = Cor @ uflat

    # Reshape result
    offset = 0
    if layout == 'mlr':
        offset = 0
        for m in range(L_max+1):
            ell_start = m if truncate else 0
            for ell in range(ell_start, L_max+1):
                ne = np.shape(uinit['c'][ell])[0]
                ucoef['c'][ell][:, m] = uflat[offset:offset+ne]
                offset += ne
    else:
        for ell in range(len(uinit['c'])):
            ne = np.prod(np.shape(ucoef['c'][ell]))
            if layout == 'lrm':
                ucoef['c'][ell] = np.reshape(uflat[offset:offset+ne],np.shape(ucoef['c'][ell]))
            else:
                ucoef['c'][ell] = np.reshape(uflat[offset:offset + ne], np.shape(ucoef['c'][ell].T)).T
            offset += ne

    # Compare solutions
    print('Computing errors...')
    print_errors(ucoef, ugrid, rank=1, plot=False)
    print('Done!')
    return ucoef


if __name__=='__main__':
    ucoef_lrm = test_coriolis('lrm')
    ucoef_lmr = test_coriolis('lmr')
    ucoef_mlr = test_coriolis('mlr', truncate=False)
    ucoef_mlr_t = test_coriolis('mlr', truncate=True)

    print('Comparing lrm vs lmr...')
    print_errors(ucoef_lrm, ucoef_lmr, rank=1, plot=False, tol=0.)
    print('Comparing lrm vs mlr...')
    print_errors(ucoef_lrm, ucoef_mlr, rank=1, plot=False, tol=1e-16)
    print('Comparing mlr (full) vs mlr (truncated)...')
    print_errors(ucoef_mlr, ucoef_mlr_t, rank=1, plot=False, tol=0.)

    print('Done!')

