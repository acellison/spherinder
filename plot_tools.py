from dedalus_sphere import ball_wrapper as ball
from dedalus.extras import plot_tools
from mpi4py import MPI
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import MaxNLocator
import dedalus.public as de
from dedalus.core.distributor import Distributor


def create_axes(num, aspect=None, plot_kwargs=None, make_cbar=True):
    if aspect is None:
        aspect = 1
    if plot_kwargs is None:
        plot_kwargs = {}

    t_mar, b_mar, l_mar, r_mar = (0.08, 0.02, 0.02, 0.02)
    h_plot, w_plot = (1, 1)
    w_pad = 0.08
    h_pad = 0.05
    h_cbar = 0.03

    if not make_cbar:
        t_mar = 0.04
        h_cbar = 0.

    h_total = t_mar + h_plot + h_pad + h_cbar + b_mar
    w_total = l_mar + num * w_plot + (num-1) * w_pad + r_mar

    width = 6
    scale = width / w_total

    fig = plt.figure(figsize=(aspect * scale * w_total, scale * h_total))

    plot_axes = []
    for i in range(num):
        left = (l_mar + (w_pad + w_plot) * i) / w_total
        bottom = 1 - (t_mar + h_plot + h_pad + h_cbar) / h_total
        width = w_plot / w_total
        height = h_plot / h_total
        plot_axes.append(fig.add_axes([left, bottom, width, height], **plot_kwargs))
    if num == 1:
        plot_axes = plot_axes[0]

    if not make_cbar:
        return fig, plot_axes

    cbar_axes = []
    for i in range(num):
        left = (l_mar + (w_pad + w_plot) * i) / w_total
        bottom = 1 - (t_mar + h_cbar) / h_total
        width = w_plot / w_total
        height = h_cbar / h_total
        cbar_axes.append(fig.add_axes([left, bottom, width, height]))
    if num == 1:
        cbar_axes = cbar_axes[0]

    return fig, plot_axes, cbar_axes


def plotsurface(theta,phi,u):
    theta = theta.ravel()
    phi = phi.ravel()
    u = u.ravel()

    theta, phi = np.meshgrid(theta, phi)
    theta, phi = theta.ravel(), phi.ravel()
    mesh_x, mesh_y = theta * np.cos(phi), theta * np.sin(phi)
    triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
    x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    colors = np.mean(u[triangles], axis=1)

    # Set up plot
    fig, plot_axes, cbar_axes = create_axes(1, aspect=1.2, plot_kwargs={'projection':'3d'})

    # Plot the surface
    cmap = plt.get_cmap('RdBu')
    triang = mtri.Triangulation(x, y, triangles)
    collec = plot_axes.plot_trisurf(triang, z, cmap=cmap, shade=False, linewidth=0., edgecolor=None, antialiased=False)
    collec.set_array(colors)
    collec.autoscale()
    plot_axes.set_xlabel('x')
    plot_axes.set_ylabel('y')
    plot_axes.set_zlabel('z')

    # Plot the colorbar
    nbins = 7
    cbar = fig.colorbar(collec, cax=cbar_axes, orientation='horizontal', ticks=MaxNLocator(nbins=nbins))
    cbar.ax.tick_params(labelsize=8)

    return fig, plot_axes


def plotequatorialquiver(ux, uy, r, theta, phi):
    num = 1
    fig, plot_axes = create_axes(num, make_cbar=False)

    # find the two nearest theta values to the equator
    angle = np.pi/2
    inds = np.argpartition(abs(theta.ravel()-angle), 2)[:2]

    phi_shape = phi.shape
    r_shape = r.shape
    phi = phi.reshape((phi_shape[0], 1))
    r = r.reshape((1, r_shape[-1]))

    # Average the data at the two nearest theta values to yield theta = pi/2
    # Fixme: linear interpolate instead
    if theta[0,inds[0],0] == angle: inds[1] = inds[0]
    vx = 0.5*(ux[:,inds[0],:] + ux[:,inds[1],:])
    vy = 0.5*(ux[:,inds[0],:] + uy[:,inds[1],:])

    r, phi = np.meshgrid(r.ravel(), phi.ravel())
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    eps = 0.02
    plot_axes.plot((1 + eps / 2) * np.sin(phi), (1 + eps / 2) * np.cos(phi), color='k', linewidth=1)
    plot_axes.quiver(x, y, vx, vy)
    plot_axes.set_axis_off()
    plot_axes.axis([-1 - 2 * eps, 1 + 2 * eps, -1 - 2 * eps, 1 + 2 * eps])

    return fig, plot_axes


def plotmeridionalquiver(ux, uz, r, theta, phi, angle=0., cmap=None):
    num = 1
    fig, plot_axes = create_axes(num, aspect=0.5, make_cbar=False)

    theta_shape = theta.shape
    r_shape = r.shape
    theta = theta.reshape((theta_shape[1], 1))
    r = r.reshape((1, r_shape[-1]))

    phiind1 = np.argmin(abs(phi.ravel()-angle))
    vx = ux[phiind1,:,:]
    vz = uz[phiind1,:,:]

    r, theta = np.meshgrid(r.ravel(), theta.ravel())
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    eps = 0.02
    plot_axes.plot((1 + eps / 2) * np.sin(theta), (1 + eps / 2) * np.cos(theta), color='k', linewidth=1)
    plot_axes.quiver(x, y, vx, vz, scale=32)
    plot_axes.set_axis_off()
    plot_axes.axis([-2 * eps, 1 + 2 * eps, -1 - 2 * eps, 1 + 2 * eps])

    return fig, plot_axes


def plotequatorialslice(field, r, theta, phi, cmap=None):
    # find the two nearest theta values to the equator
    angle = np.pi/2
    inds = np.argpartition(abs(theta.ravel()-angle), 2)[:2]

    phi_shape = phi.shape
    r_shape = r.shape
    phi = phi.reshape((phi_shape[0], 1))
    r = r.reshape((1, r_shape[-1]))

    # Average the data at the two nearest theta values to yield theta = pi/2
    # Fixme: linear interpolate instead
    if theta[0,inds[0],0] == angle: inds[1] = inds[0]
    plot_data = 0.5*(field[:,inds[0],:] + field[:,inds[1],:])

    r, phi = plot_tools.quad_mesh(r.ravel(), phi.ravel())
    r[:, 0] = 0

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    eps = 0.02

    num = 1
    fig, plot_axes, cbar_axes = create_axes(num)

    if cmap is None:
        cmap = 'RdBu'
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='grey', alpha=.5)
    c_im = plot_axes.pcolormesh(x, y, plot_data, cmap=cmap)
    plot_axes.plot((1 + eps / 2) * np.sin(phi), (1 + eps / 2) * np.cos(phi), color='k', linewidth=1)
    plot_axes.set_axis_off()
    plot_axes.axis([-1 - 2 * eps, 1 + 2 * eps, -1 - 2 * eps, 1 + 2 * eps])
    nbins = 4
    cbar = fig.colorbar(c_im, cax=cbar_axes, orientation='horizontal', ticks=MaxNLocator(nbins=nbins))
    cbar.ax.tick_params(labelsize=8)

    return fig, plot_axes


def plotmeridionalslice(field, r, theta, phi, angle=0., stretch=False, cmap=None):
    num = 1
    fig, plot_axes, cbar_axes = create_axes(num)
    eps = 0.02
    nbins = 4

    theta_shape = theta.shape
    r_shape = r.shape
    theta = theta.reshape((theta_shape[1], 1))
    r = r.reshape((1, r_shape[-1]))

    theta = np.append(theta+np.pi, theta, axis=0)
    phiind1 = np.argmin(abs(phi.ravel()-angle))
    phiind2 = np.argmin(abs(phi.ravel()-(angle+np.pi)))
    plot_data = np.append(field[phiind2,:,:], field[phiind1,:,:], axis=0)

    theta = np.pad(theta, ((0,1),(0,0)), mode='wrap')
    theta[-1] -= 2*np.pi
    plot_data = np.pad(plot_data, ((0,1),(0,0)), mode='wrap')

    r, theta = plot_tools.quad_mesh(r.ravel(), theta.ravel())
    r[:, 0] = 0

    x = r * np.sin(theta)
    y = r * np.cos(theta)
    if stretch:
        y /= np.sqrt(1 - x**2)
        y = np.where(np.isnan(y), 0, y)

    if cmap is None:
        cmap = 'RdBu'
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color='grey', alpha=.5)
    c_im = plot_axes.pcolormesh(x, y, plot_data, cmap=cmap)

    if stretch:
        lw = 1.5
        a = 1 + eps / 2
        plot_axes.plot([-a,a], [a,a], color='k', linewidth=lw)
        plot_axes.plot([a,a], [a,-a], color='k', linewidth=lw)
        plot_axes.plot([a,-a], [-a,-a], color='k', linewidth=lw)
        plot_axes.plot([-a,-a], [-a,a], color='k', linewidth=lw)
    else:
        plot_axes.plot((1 + eps / 2) * np.sin(theta), (1 + eps / 2) * np.cos(theta), color='k', linewidth=1)

    plot_axes.set_axis_off()
    plot_axes.axis([-1 - 2 * eps, 1 + 2 * eps, -1 - 2 * eps, 1 + 2 * eps])
    cbar = fig.colorbar(c_im, cax=cbar_axes, orientation='horizontal', ticks=MaxNLocator(nbins=nbins))
    cbar.ax.tick_params(labelsize=8)

    return fig, plot_axes


def dealias(B, domain, field, L_factor, N_factor, mesh=None):
    field.require_coeff_space()

    L_dealias = L_factor
    N_dealias = N_factor

    L_max, N_max, R_max = B.L_max, B.N_max, B.R_max
    N_r = N_max
    if mesh is None:
        mesh = [1]
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # Make domain
    phi_basis2 = de.Fourier('phi', 2 * (L_max + 1), interval=(0, 2 * np.pi), dealias=L_dealias)
    theta_basis2 = de.Fourier('theta', L_max + 1, interval=(0, np.pi), dealias=L_dealias)
    r_basis2 = de.Fourier('r', N_r + 1, interval=(0, 1), dealias=N_dealias)
    domain2 = de.Domain([phi_basis2, theta_basis2, r_basis2], grid_dtype=np.float64, mesh=mesh)

    domain2.global_coeff_shape = np.array([L_max + 1, L_max + 1, N_max + 1])
    domain2.distributor = Distributor(domain2, comm, mesh)

    mesh = domain2.distributor.mesh
    if len(mesh) == 0:
        phi_layout2 = domain2.distributor.layouts[3]
        th_m_layout2 = domain2.distributor.layouts[2]
        ell_r_layout2 = domain2.distributor.layouts[1]
        r_ell_layout2 = domain2.distributor.layouts[1]
    elif len(mesh) == 1:
        phi_layout2 = domain2.distributor.layouts[4]
        th_m_layout2 = domain2.distributor.layouts[2]
        ell_r_layout2 = domain2.distributor.layouts[1]
        r_ell_layout2 = domain2.distributor.layouts[1]
    elif len(mesh) == 2:
        phi_layout2 = domain2.distributor.layouts[5]
        th_m_layout2 = domain2.distributor.layouts[3]
        ell_r_layout2 = domain2.distributor.layouts[2]
        r_ell_layout2 = domain2.distributor.layouts[1]

    m_start = th_m_layout2.slices(scales=1)[0].start
    m_end = th_m_layout2.slices(scales=1)[0].stop - 1
    m_size = m_end - m_start + 1
    ell_start = r_ell_layout2.slices(scales=1)[1].start
    ell_end = r_ell_layout2.slices(scales=1)[1].stop - 1

    # set up ball
    N_theta = int((L_max + 1) * L_dealias)
    N_r = int((N_r + 1) * N_dealias)
    B2 = ball.Ball(N_max, L_max, N_theta=N_theta, N_r=N_r, R_max=R_max, ell_min=ell_start, ell_max=ell_end,
                   m_min=m_start, m_max=m_end, a=0.)
    theta_global = B2.grid(0)
    r_global = B2.grid(1)
    phi_global = domain.grid(0, scales=domain2.dealias)

    field2 = ball.TensorField_3D(field.rank, B2, domain2)
    for ell in range(ell_start, ell_end + 1):
        ell_local = ell - ell_start
        field2['c'][ell_local] = field['c'][ell_local]

    r_global = np.reshape(r_global, (1, 1, np.prod(np.shape(r_global))))
    theta_global = np.reshape(theta_global, (1, np.prod(np.shape(theta_global)), 1))

    return field2, r_global, theta_global, phi_global


def sph2cart(u, theta, phi):
    ur, utheta, uphi = u['g'][0], u['g'][1], u['g'][2]
    ux = ur * np.sin(theta) * np.cos(phi) + utheta * np.cos(theta) * np.cos(phi) - uphi * np.sin(phi)
    uy = ur * np.sin(theta) * np.sin(phi) + utheta * np.cos(theta) * np.sin(phi) + uphi * np.cos(phi)
    uz = ur * np.cos(theta) - utheta * np.sin(theta)
    return ux, uy, uz
