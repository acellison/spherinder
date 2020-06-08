from dedalus_sphere import ball_wrapper as ball
import dedalus.public as de
from dedalus.core.distributor import Distributor
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from plot_tools import create_axes
from matplotlib.ticker import MaxNLocator
from interpolate import make_xz_mesh, sph2cart, interpolate
import scipy.fft as fft


# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Resolution
L_max = 63 
N_max = 63
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


def plotmeridionalslice(field, x, z, stretch=False, cmap=None):
    num = 1
    fig, plot_axes, cbar_axes = create_axes(num)
    eps = 0.02
    nbins = 4

    if stretch:
        z /= np.sqrt(1 - x**2)
        z = np.where(np.isnan(z), 0, z)

    if cmap is None:
        cmap = 'RdBu'
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='grey', alpha=.5)
    c_im = plot_axes.pcolormesh(x, z, field, cmap=cmap)

    plot_axes.set_axis_off()
    plot_axes.axis([-1 - 2 * eps, 1 + 2 * eps, -1 - 2 * eps, 1 + 2 * eps])
    cbar = fig.colorbar(c_im, cax=cbar_axes, orientation='horizontal', ticks=MaxNLocator(nbins=nbins))
    cbar.ax.tick_params(labelsize=8)


def chebgrid(n):
    return np.cos((np.arange(0,n+1)+0.5)*np.pi/(n+1))


def test_interpolate_cheb():
    T = ball.TensorField_3D(0, B, domain)
    T['g'] = 0.5 * (1 - r ** 2) + 0.1 / 8. * np.sqrt(35 / np.pi) * r ** 3 * (1 - r ** 2) * (
                np.cos(3 * phi) + np.sin(3 * phi)) * np.sin(theta) ** 3

    # Convert from spherical to cartesian grid on the phi = 0 axis (y = 0)
    print('  (r, θ, φ) -> (x, y, z)...')

    # Create evaluation points
    nz = 128
    x_eval, z_eval = 0.5, chebgrid(nz)
    z_eval *= (1 - x_eval**2)**.5 * .998
    xi = np.array([[x_eval, z] for z in z_eval])

    # Interpolate
    zresult = interpolate(T, r, theta, phi, xi)
    dctz = fft.dct(zresult)

    # Create evaluation points
    nx = 128
    x_eval, z_eval = chebgrid(nx), 0.
    x_eval *= (1 - z_eval**2)**.5 * .998
    xi = np.array([[x, z_eval] for x in x_eval])

    # Interpolate
    xresult = interpolate(T, r, theta, phi, xi)
    dctx = fft.dct(xresult)

    plt.figure()
    plt.semilogy(abs(dctz)/max(abs(dctz)))
    plt.semilogy(abs(dctx)/max(abs(dctx)))
    plt.grid()


def test_interpolate_2d():
    # initial condition
    T = ball.TensorField_3D(0, B, domain)
    T['g'] = 0.5 * (1 - r ** 2) + 0.1 / 8. * np.sqrt(35 / np.pi) * r ** 3 * (1 - r ** 2) * (
                np.cos(3 * phi) + np.sin(3 * phi)) * np.sin(theta) ** 3
    field = T

    # Convert from spherical to cartesian grid on the phi = 0 axis (y = 0)
    print('  (r, θ, φ) -> (x, y, z)...')
    points_0 = np.array([sph2cart(rr, tt, 0.)    for tt in theta.ravel() for rr in r.ravel()]) 
    points_1 = np.array([sph2cart(rr, tt, np.pi) for tt in theta.ravel() for rr in r.ravel()]) 

    # Remove y coordinate
    points_0 = np.array([[point[0], point[2]] for point in points_0])
    points_1 = np.array([[point[0], point[2]] for point in points_1])
    points = np.append(points_0, points_1, axis=0)

    # Get field values at phi = 0
    nphi = np.shape(field['g'][0])[0]
    signal = np.append(field['g'][0][0], field['g'][0][nphi//2], axis=0)

    # Create evaluation points
    nx, nz = 128, 128
    x_eval, z_eval = make_xz_mesh(nx, nz)
    xi = np.array([x_eval.ravel(), z_eval.ravel()]).T

    # Interpolate
    result = interpolate(field, r, theta, phi, xi)

    # Plot original signal
    x, z = np.reshape(points[:,0], np.shape(signal)), np.reshape(points[:,1], np.shape(signal))
    plotmeridionalslice(signal, x, z)

    # Plot the new funky
    result = np.reshape(result, (nx, nz))
    x, z = x_eval, z_eval
    plotmeridionalslice(result, x, z)


if __name__=='__main__':
#    test_interpolate_2d()
    test_interpolate_cheb()
    plt.show()


