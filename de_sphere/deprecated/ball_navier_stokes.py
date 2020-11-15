from dedalus_sphere import ball_wrapper as ball
from dedalus_sphere import ball128
import numpy as np
from   scipy.linalg      import eig
from scipy.sparse        import linalg as spla
import scipy.sparse      as sparse
import scipy.special     as spec
import dedalus.public as de
from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.core.distributor import Distributor
from mpi4py import MPI
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
from dedalus_sphere import timesteppers
import boussinesq
import state_vector.StateVector as StateVec

import logging
logger = logging.getLogger(__name__)

implicit_buoyancy = False
implicit_base_temp = False
use_deviation_temp = False
spectral_coriolis = False

class StateVector:

    def __init__(self,u,p,T):
        self.data = []
        for ell in range(ell_start,ell_end+1):
            if ell == 0: taus = np.zeros(1)
            else: taus = np.zeros(4)
            ell_local = ell-ell_start
            for m in range(m_start,m_end+1):
                m_local = m-m_start
                self.data.append(np.concatenate((u['c'][ell_local][:,m_local],p['c'][ell_local][:,m_local],
                                                 T['c'][ell_local][:,m_local],taus)))

    def pack(self,u,p,T):
        for ell in range(ell_start,ell_end+1):
            if ell == 0: taus = np.zeros(1)
            else: taus = np.zeros(4)
            ell_local = ell-ell_start
            for m in range(m_start,m_end+1):
                m_local = m-m_start
                self.data[ell_local*m_size+m_local] = np.concatenate((u['c'][ell_local][:,m_local],
                                                                      p['c'][ell_local][:,m_local],
                                                                      T['c'][ell_local][:,m_local],
                                                                      taus))

    def unpack(self,u,p,T):
        u.layout = 'c'
        p.layout = 'c'
        T.layout = 'c'
        for ell in range(ell_start,ell_end+1):
            ell_local = ell-ell_start
            end_u = u['c'][ell_local].shape[0]
            p_len = p['c'][ell_local].shape[0]
            for m in range(m_start,m_end+1):
                m_local = m - m_start
                u['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][:end_u]
                p['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][end_u:end_u+p_len]
                T['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][end_u+p_len:end_u+2*p_len]


# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Resolution
L_max = 31
N_max = 31
R_max = 3

alpha_BC = 0

#L_dealias = 3/2
#N_dealias = 3/2
L_dealias = 1
N_dealias = 1
N_r = N_max

# parameters
Ekman = 3e-4
Prandtl = 1
Rayleigh = 95
S = 3

# Integration parameters
dt = 2e-5
t_end = 20

# Make domain
# mesh=[4,4]
mesh=[4,1]
# mesh=[1]
phi_basis = de.Fourier('phi',2*(L_max+1), interval=(0,2*np.pi),dealias=L_dealias)
theta_basis = de.Fourier('theta', L_max+1, interval=(0,np.pi),dealias=L_dealias)
r_basis = de.Fourier('r', N_max+1, interval=(0,1),dealias=N_dealias)
domain = de.Domain([phi_basis,theta_basis,r_basis], grid_dtype=np.float64, mesh=mesh)

domain.global_coeff_shape = np.array([L_max+1,L_max+1,N_max+1])
domain.distributor = Distributor(domain,comm,mesh)

mesh = domain.distributor.mesh
if len(mesh) == 0:
    phi_layout   = domain.distributor.layouts[3]
    th_m_layout  = domain.distributor.layouts[2]
    ell_r_layout = domain.distributor.layouts[1]
    r_ell_layout = domain.distributor.layouts[1]
elif len(mesh) == 1:
    phi_layout   = domain.distributor.layouts[4]
    th_m_layout  = domain.distributor.layouts[2]
    ell_r_layout = domain.distributor.layouts[1]
    r_ell_layout = domain.distributor.layouts[1]
elif len(mesh) == 2:
    phi_layout   = domain.distributor.layouts[5]
    th_m_layout  = domain.distributor.layouts[3]
    ell_r_layout = domain.distributor.layouts[2]
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

weight_theta = B.weight(1,dimensions=3)[:,grid_slices[1],:]
weight_r = B.weight(2,dimensions=3)[:,:,grid_slices[2]]

om = ball.TensorField_3D(1,B,domain)
u  = ball.TensorField_3D(1,B,domain)
p  = ball.TensorField_3D(0,B,domain)
T  = ball.TensorField_3D(0,B,domain)
DT = ball.TensorField_3D(1,B,domain)

u_rhs = ball.TensorField_3D(1,B,domain)
p_rhs = ball.TensorField_3D(0,B,domain)
T_rhs = ball.TensorField_3D(0,B,domain)

print([np.shape(u['c'][ell]) for ell in range(len(u['c']))])

# Initial temperature condition
needs_base_temp = implicit_base_temp or use_deviation_temp
if needs_base_temp:
    Tb = 0.5*(1-r**2)
    T['g'] = 0.1/8.*np.sqrt(35/np.pi)*r**3*(1-r**2)*(np.cos(3*phi)+np.sin(3*phi))*np.sin(theta)**3
else:
    T['g'] = 0.5*(1-r**2) + 0.1/8.*np.sqrt(35/np.pi)*r**3*(1-r**2)*(np.cos(3*phi)+np.sin(3*phi))*np.sin(theta)**3

# build state vector
state_vector = StateVector(u,p,T)
NL = StateVector(u,p,T)
timestepper = timesteppers.SBDF4(StateVector, u,p,T)

# build matrices
M,L,P,LU = [],[],[],[]
for ell in range(ell_start,ell_end+1):
    N = B.N_max - B.N_min(ell-B.R_max)
    M_ell,L_ell = boussinesq.matrices(B,N,ell,Ekman,Prandtl,Rayleigh,alpha_BC,implicit_buoyancy,implicit_base_temp)
    M.append(M_ell.astype(np.complex128))
    L.append(L_ell.astype(np.complex128))
    P.append(M_ell.astype(np.complex128))
    LU.append([None])


def create_u_dot_r_matrices():
    def R(N, ell, mu, i, deg):
        """Multiplication by r"""
        if mu == +1: return B.op('R+', N, i, ell+deg)
        if mu == -1: return B.op('R-', N, i, ell+deg)

    Lm, Lp = [], []
    for ell in range(ell_start, ell_end + 1):
        N = B.N_max - B.N_min(ell - B.R_max)
        if ell == 0:
            xim, xip = 0, 1
        else:
            xim, xip = B.xi([-1, +1], ell)
        Lm.append(xim * R(N, ell, +1, 0, -1))
        Lp.append(xip * R(N, ell, -1, 0, +1))
    return Lm, Lp


if use_deviation_temp:
    L_udotr_m, L_udotr_p = create_u_dot_r_matrices()


def compute_u_dot_r(uin, scale, result):
    """Compute u.r in spectral space, adding into result after scaling"""
    for ell in range(ell_start, ell_end + 1):
        ell_local = ell - ell_start

        u_len = uin['c'][ell_local].shape[0]
        um = uin['c'][ell_local][:u_len//3, :]
        up = uin['c'][ell_local][2*u_len//3:, :]

        # Spectral space multiplication by r
        rum = L_udotr_m[ell_local] @ um
        rup = L_udotr_p[ell_local] @ up
        result['c'][ell_local] += scale * rum
        result['c'][ell_local] += scale * rup


if spectral_coriolis:
    # Note: Coriolis couples ell so we cannot parallelize in this dimension.  m remains decoupled
    sv = StateVec(B, layout='lrm', fields=[('u',1)], ntau=lambda ell: 0, truncate=False)
    Cor = boussinesq.create_coriolis_matrix(B, sv)
    uflat = np.zeros(np.shape(Cor)[0], dtype=u['c'][0].dtype)


def compute_coriolis(uin, uflat, scale, result):
    offset = 0
    for ell in range(len(uin['c'])):
        ne = np.prod(np.shape(uin['c'][ell]))
        uflat[offset:offset+ne] = uin['c'][ell].ravel()
        offset += ne

    if offset != len(uflat):
        raise ValueError('Incorrect u dimensions!  Parallelization in ell not possible with spectral space Coriolis.')

    # Apply Coriolis
    uflat = Cor @ uflat

    # Reshape result
    offset = 0
    for ell in range(len(uin['c'])):
        ne = np.prod(np.shape(uin['c'][ell]))
        result['c'][ell] += scale*np.reshape(uflat[offset:offset+ne], np.shape(uin['c'][ell]))
        offset += ne


# calculate RHS terms from state vector
def nonlinear(state_vector, RHS, t):

    # get U in coefficient space
    state_vector.unpack(u,p,T)

    DT.layout = 'c'
    om.layout = 'c'
    # take derivatives
    for ell in range(ell_start,ell_end+1):
        ell_local = ell - ell_start
        B.curl(ell,1,u['c'][ell_local],om['c'][ell_local])
        DT['c'][ell_local] = B.grad(ell,0,T['c'][ell_local])

    u_rhs.layout = 'g'
    u_rhs['g'] = Ekman * B.cross_grid(u['g'], om['g'])

    if implicit_base_temp:
        if implicit_buoyancy:
            # Fully implicit linearized temperature
            # Do nothing - pressure should balance the temperature base state anyway
            pass
        else:
            # Linearized temperature, explicit buoyancy term.  Compute the full temperature profile
            u_rhs['g'][0] += Rayleigh * r * (T['g'][0] + Tb)
    else:
        if implicit_buoyancy:
            # Full temperature profile, implicit buoyancy term.  No contribution to RHS
            pass
        else:
            # Full temperature profile, explicit buoyancy.  Full contribution to RHS
            u_rhs['g'][0] += Rayleigh * r * T['g'][0]

    # Spectral coriolis done here to minimize number of transposes
    if spectral_coriolis:
        compute_coriolis(u, uflat, -1., u_rhs)
    else:
        ez = np.array([np.cos(theta),-np.sin(theta),0*np.cos(theta)])
        u_rhs['g'] -= B.cross_grid(ez, u['g'])

    # Advect the temperature!   Base temperature balances S so its contributions are handled
    # without explicitly adding it in to the RHS
    T_rhs.layout = 'g'
    if needs_base_temp:
        T_rhs['g'] = -Prandtl*(u['g'][0]*DT['g'][0] + u['g'][1]*DT['g'][1] + u['g'][2]*DT['g'][2])  # -Pr (u.Grad) T
    else:
        T_rhs['g'] = S - Prandtl*(u['g'][0]*DT['g'][0] + u['g'][1]*DT['g'][1] + u['g'][2]*DT['g'][2])  # S - Pr (u.Grad) T

    # Explicit Pr u.r (gradient of base temperature) term computed in spectral space
    if use_deviation_temp and not implicit_base_temp:
        compute_u_dot_r(u, Prandtl, T_rhs)

    # transform (ell, r) -> (ell, N)
    for ell in range(ell_start, ell_end+1):
        ell_local = ell - ell_start

        # multiply by conversion matrices (may be very important)
        # note that M matrices are no longer conversion matrices -- need to divide by Ekman or Prandtl
        u_len = u_rhs['c'][ell_local].shape[0]
        u_rhs['c'][ell_local] = M[ell_local][:u_len,:u_len].dot(u_rhs['c'][ell_local])/Ekman
        p_len = p_rhs['c'][ell_local].shape[0]
        T_rhs['c'][ell_local] = M[ell_local][u_len+p_len:u_len+2*p_len,u_len+p_len:u_len+2*p_len].dot(T_rhs['c'][ell_local])/Prandtl

    RHS.pack(u_rhs,p_rhs,T_rhs)


reducer = GlobalArrayReducer(domain.dist.comm_cart)

t = 0.

t_list = []
E_list = []

# timestepping loop
start_time = time.time()
iter = 0

while t < t_end:
    nonlinear(state_vector,NL,t)

    if iter % 10 == 0:
        E0 = np.sum(weight_r*weight_theta*0.5*u['g']**2)*(np.pi)/((L_max+1)*L_dealias)
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        logger.info("iter: {:d}, dt={:e}, t/t_e={:e}, E0={:1.18e}".format(iter, dt, t/t_end,E0))
        if rank == 0:
            t_list.append(t)
            E_list.append(E0)

    timestepper.step(dt, state_vector, B, L, M, P, NL, LU)
    t += dt
    iter += 1

end_time = time.time()
if rank==0:
    print('simulation took: %f' %(end_time-start_time))
    t_list = np.array(t_list)
    E_list = np.array(E_list)
    np.savetxt('marti_conv.dat',np.array([t_list,E_list]))

