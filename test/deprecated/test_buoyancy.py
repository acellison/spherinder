from dedalus_sphere import ball_wrapper as ball
import dedalus.public as de
from dedalus.core.distributor import Distributor
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt


# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Resolution
L_max = 63
N_max = 63
#L_max = 127
#N_max = 127
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


def print_errors(coef, grid, rank=0, plot=False):
    if plot == False or plot is None:
        plot = -1
    # Compare in coefficient space
    for i in range(len(grid['c'])):
        rawerror = coef['c'][i]-grid['c'][i]
        error = abs(rawerror)
        maxerror = np.max(error)
        if maxerror > 1e-16:
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
        if maxerror > 1e-16:
            print('Grid space: ind: {}, maxerror: {}, L1error: {}'.format(ind, maxerror, l1error))


def test_buoyancy():
    print('testing buoyancy term...')
    def R(N, ell, mu, i, deg):
        """Multiplication by r"""
        if mu == +1: return B.op('R+', N, i, ell+deg)
        if mu == -1: return B.op('R-', N, i, ell+deg)

    ugrid = ball.TensorField_3D(1,B,domain)
    ucoef = ball.TensorField_3D(1,B,domain)
    Ttest = ball.TensorField_3D(0,B,domain)
    Ttest.layout = 'g'

    # initial condition
    T = ball.TensorField_3D(0, B, domain)
    T['g'] = 0.5 * (1 - r ** 2) + 0.1 / 8. * np.sqrt(35 / np.pi) * r ** 3 * (1 - r ** 2) * (
                np.cos(3 * phi) + np.sin(3 * phi)) * np.sin(theta) ** 3

    Ttest['g'][0] = T['g'][0] + 1/(1+r**2) + 1/(2+np.sin(theta)) + 1/(2+np.sin(phi/2))

    # Convert to coefficient space and back for fair comparisons
    Ttest['c']
    Ttest['g']

    # Compute the buoyancy term on the grid
    print('Computing buoyancy on the grid...')
    ugrid.layout = 'g'
    ugrid['g'][0] = r*Ttest['g'][0]  # Ra T r

    # Compute the buoyancy term in coefficient space
    print('Computing buoyancy in coefficient space...')
    ucoef.layout = 'c'
    for ell in range(ell_start, ell_end+1):
        N = B.N_max - B.N_min(ell-B.R_max)
        ell_local = ell - ell_start

        L24 = R(N, ell, +1, 0, 0)
        if ell == 0:
            L04 = np.zeros(np.shape(L24))  # Can't decrement from ell = 0!
            xim, xip = 1, 1
        else:
            L04 = R(N, ell, -1, 0, 0)
            xim, xip = B.xi([-1, +1], ell)

        # Spectral space multiplication by r
        u_len = ucoef['c'][ell_local].shape[0]
        ucoef['c'][ell_local][:u_len//3, :] = xim*L04.dot(Ttest['c'][ell_local])
        ucoef['c'][ell_local][2*u_len//3:, :] = xip*L24.dot(Ttest['c'][ell_local])

    print_errors(ucoef, ugrid, rank=1, plot=False)


def test_rmult():
    """Test multiplication of the radial component of a field by r.  
       In the temperature equation we require the term r * u_r to linearize about the base temperature 
       Tb = 0.5*(1-r**2) with gradient Grad Tb = -r e_r.  Then the advective term becomes 
       u.Grad Tb = -r * u_r, with u_r the radial component of the velocity field"""
    print('testing u.r term...')
    def R(N, ell, mu, i, deg):
        """Multiplication by r"""
        if mu == +1: return B.op('R+', N, i, ell+deg)
        if mu == -1: return B.op('R-', N, i, ell+deg)

    random_initial_condition = False
    ugrid = ball.TensorField_3D(1, B, domain)
    Tgrid = ball.TensorField_3D(0, B, domain)
    Tcoef = ball.TensorField_3D(0, B, domain)

    if random_initial_condition:
        ugrid.layout = 'c'
        for ell in range(len(ugrid['c'])):
            rows = np.shape(ugrid['c'][ell])[0]
            cols = ell+1
            ugrid['c'][ell][:rows,:cols] += np.random.random((rows,cols))
    else:
        ugrid.layout = 'g'
        ugrid['g'][0] = r*(np.cos(theta)**2 - r**2*np.cos(phi)*np.cos(theta)**3*np.sin(theta) + np.cos(phi)*(-1 + r*np.cos(theta))*np.sin(phi)*np.sin(theta)**2 +
             - r*np.sin(phi)**3*np.sin(theta)**3 + r**7*np.cos(phi)**8*np.cos(theta)*np.sin(theta)**8)
        ugrid['g'][1] = -  -(r*(r**7*np.cos(phi)**8*np.sin(theta)**9 + np.cos(phi)*np.cos(theta)*(r**2*np.cos(theta)**3 + (1 - r*np.cos(theta))*np.sin(phi)*np.sin(theta)) +
             - np.cos(theta)*np.sin(theta)*(1 - r*np.sin(phi)**3*np.sin(theta))))
        ugrid['g'][2] = -  r*(r**2*np.cos(theta)**3*np.sin(phi) + r*np.cos(phi)**2*np.cos(theta)*np.sin(theta) + np.sin(phi)**2*np.sin(theta)*(1 + r*np.cos(phi)*np.sin(theta)))
        ugrid['c']
        ugrid['g']

    # Convert ugrid to coefficient space and back
    ugrid['g']
    ugrid['c']

    # Compute r multiplication on the grid
    print('Computing r multiplication on the grid...')
    Tgrid.layout = 'g'
    Tgrid['g'][0] = r * ugrid['g'][0]
    Tgrid['c']

    # Compute r multiplication in coefficient space
    print('Computing r multiplication in coefficient space...')
    Tcoef.layout = 'c'
    for ell in range(ell_start, ell_end + 1):
        N = B.N_max - B.N_min(ell - B.R_max)
        ell_local = ell - ell_start

        if ell == 0:
            xim, xip = 0, 1
        else:
            xim, xip = B.xi([-1, +1], ell)
        Npad = 0
        Ntot = N+Npad
        L40 = xim * R(Ntot, ell, +1, 0, -1)
        L42 = xip * R(Ntot, ell, -1, 0, +1)

        u_len = ugrid['c'][ell_local].shape[0]
        um = ugrid['c'][ell_local][:u_len//3, :]
        up = ugrid['c'][ell_local][2*u_len//3:, :]
        um = np.pad(um, [(0,Npad),(0,0)],mode='constant', constant_values=0)
        up = np.pad(up, [(0,Npad),(0,0)],mode='constant', constant_values=0)

        # Spectral space multiplication by r
        rum = L40 @ um
        rup = L42 @ up
        Tcoef['c'][ell_local] += rum[:u_len//3, :]
        Tcoef['c'][ell_local] += rup[:u_len//3, :]

    print_errors(Tcoef, Tgrid, rank=0, plot=False)


if __name__=='__main__':
    test_buoyancy()
    test_rmult()


