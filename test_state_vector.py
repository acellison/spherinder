from dedalus_sphere import ball_wrapper as ball
import dedalus.public as de
from dedalus.core.distributor import Distributor
from mpi4py import MPI
import numpy as np
from state_vector import StateVector


# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Resolution
L_max = 31
N_max = 31
R_max = 3

L_dealias = 1
N_dealias = 1
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


def test_state_vector(layout, m_min=None, m_max=None, truncate=True):
    print('testing state vector construction, layout = {}, truncate = {}, (m_min, m_max) = ({}, {})...'.format(layout, truncate, m_min, m_max))

    m_min = B.m_min if m_min is None else m_min
    m_max = B.m_min if m_max is None else m_max

    u1 = ball.TensorField_3D(1,B,domain)
    p1 = ball.TensorField_3D(0,B,domain)
    u1full = ball.TensorField_3D(1,B,domain)
    p1full = ball.TensorField_3D(0,B,domain)

    random_initial_condition = True
    if random_initial_condition:
        np.random.seed(0)
        u1.layout = 'c'
        p1.layout = 'c'
        for ell in range(B.L_max+1):
            nrowsu = np.shape(u1['c'][ell])[0]
            nrowsp = np.shape(p1['c'][ell])[0]
            nc = ell+1
            u1['c'][ell][:nrowsu,:nc] += np.random.random((nrowsu,nc))
            p1['c'][ell][:nrowsp, :nc] += np.random.random((nrowsp, nc))
            u1full['c'][ell] = u1['c'][ell]  # Attempt to spoof m-limited state_vectors
            p1full['c'][ell] = p1['c'][ell]  # Attempt to spoof m-limited state_vectors

        if m_min > 0:
            for ell in range(B.L_max+1):
                nrowsu = np.shape(u1['c'][ell])[0]
                nrowsp = np.shape(p1['c'][ell])[0]
                nc = m_min
                u1['c'][ell][:nrowsu,:nc] = np.zeros((nrowsu,nc), dtype=np.complex128)
                p1['c'][ell][:nrowsp,:nc] = np.zeros((nrowsp,nc), dtype=np.complex128)
        if m_max < B.L_max:
            for ell in range(B.L_max+1):
                nrowsu = np.shape(u1['c'][ell])[0]
                nrowsp = np.shape(p1['c'][ell])[0]
                nc = B.L_max-m_max
                u1['c'][ell][:nrowsu,m_max+1:] = np.zeros((nrowsu,nc), dtype=np.complex128)
                p1['c'][ell][:nrowsp,m_max+1:] = np.zeros((nrowsp,nc), dtype=np.complex128)

    ntau = lambda ell: 1 if ell == 0 else 4
    state_vector = StateVector(B, layout, [('u', 1), ('p', 0)], ntau=ntau, m_min=m_min, m_max=m_max, truncate=truncate)

    v = state_vector.pack([u1full, p1full])

    u2 = ball.TensorField_3D(1,B,domain)
    p2 = ball.TensorField_3D(0,B,domain)
    state_vector.unpack(v, [u2, p2])

    for ell in range(B.L_max):
        assert np.all(u2['c'][ell] == u1['c'][ell])
        assert np.all(p2['c'][ell] == p1['c'][ell])


if __name__=='__main__':
    layouts = ['lrm', 'lmr', 'mlr']
    for layout in layouts:
        test_state_vector(layout, truncate=False)

    for layout in layouts:
        test_state_vector(layout, truncate=True)

    for layout in layouts:
        test_state_vector(layout, m_min=3, m_max=8, truncate=False)

    for layout in layouts:
        test_state_vector(layout, m_min=3, m_max=8, truncate=True)

    for layout in layouts:
        test_state_vector(layout, m_min=5, m_max=5, truncate=True)
