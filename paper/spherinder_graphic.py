import os
import numpy as np
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'


def plot_surface(ax, x, y, z, facecolors, alpha=1.0, edgecolor='none', **kwargs):
    edgecolor = kwargs.pop('edgecolor', 'none')
    antialiased = kwargs.pop('antialiased', False)
    ax.plot_surface(x, y, z, facecolors=facecolors, rstride=1, cstride=1, alpha=alpha, linewidth=0, edgecolor=edgecolor, antialiased=antialiased)


def plot_cylinder(ax, nphi, nz, cmap, zscale=1., radius=1., x0=0., y0=0., alpha=1.):
    phi, z = np.mgrid[0:2*np.pi:1j*nphi, -1:1:1j*nz]
    z *= zscale
    x, y = radius*np.cos(phi) + x0, radius*np.sin(phi) + y0
    plot_surface(ax, x, y, z, facecolors=cmap(0*x*y+0.5), alpha=alpha, antialiased=True)


def main():
    m, l = 2, 3
    nphi, ntheta = 512, 256

    # Create the mesh
    equispaced_theta = True
    if equispaced_theta:
        phi, theta = np.mgrid[0:2*np.pi:1j*nphi, 0:np.pi:1j*ntheta]
        x = np.cos(phi)*np.sin(theta)
        y = np.sin(phi)*np.sin(theta)
        z = np.cos(theta)
    else:
        phi, z = np.mgrid[0:2*np.pi:1j*nphi, 1:-1:1j*ntheta]
        theta = np.arccos(z)
        x = np.cos(phi)*np.sin(theta)
        y = np.sin(phi)*np.sin(theta)

    # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
    fcolors = sph_harm(m, l, phi, theta).real
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin)/(fmax - fmin)

    same_figure = True
    if True:
        margin = 0.
        cmap = cm.RdBu
    else:
        margin = 0.15
        cmap = cm.seismic
    fcolors = (1-margin)*fcolors + margin/2

    def prepare_axes(ax, zlabel):
        fontsize = 18
        ticks = np.linspace(-1,1,5)
        ax.set_xlabel('$x$', fontsize=fontsize)
        ax.set_ylabel('$y$', fontsize=fontsize)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(zlabel, fontsize=fontsize)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
        ax.set_box_aspect([1,1,1])

    def save(fig, filename):
        filename = os.path.abspath(os.path.join('./figures/spherinder_graphic', filename))
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        fig.savefig(filename, dpi=300)

    # Create the figure
    if same_figure:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        pos1, pos2 = 121, 122
    else:
        fig = plt.figure(figsize=plt.figaspect(1.))
        pos1, pos2 = (111,)*2

    # Plot the spherical harmonic on a sphere
    ax1 = fig.add_subplot(pos1, projection='3d')
    plot_surface(ax1, x, y, z, facecolors=cmap(fcolors))
    prepare_axes(ax1, '$z$')
    if not same_figure:
        save(fig, 'spherinder_mapping-cartesian.png')

    azim, elev = ax1.azim, ax1.elev
    xlim, ylim, zlim = ax1.get_xlim3d(), ax1.get_ylim3d(), ax1.get_zlim3d()

    # Plot the spherical harmonic on a sphere
    if not same_figure:
        fig = plt.figure(figsize=plt.figaspect(1.))
    ax2 = fig.add_subplot(pos2, projection='3d')
    plot_surface(ax2, x[:,:ntheta//2], y[:,:ntheta//2], 0*z[:,:ntheta//2]+1.0,  facecolors=cmap(  fcolors[:,:ntheta//2]))
    plot_surface(ax2, x[:,:ntheta//2], y[:,:ntheta//2], 0*z[:,:ntheta//2]-1.0,  facecolors=cmap(1-fcolors[:,:ntheta//2]))  # Hack for (s,z)==(0,-1)
    plot_cylinder(ax2, nphi, nz=ntheta, cmap=cmap, zscale=0.99, alpha=0.4)
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_xlim3d(xlim)
    ax2.set_ylim3d(ylim)
    ax2.set_zlim3d(zlim)
    prepare_axes(ax2, r'$\eta$')
    fig.set_tight_layout(True)

    if not same_figure:
        filename = 'spherinder_mapping-stretched.png'
    else:
        filename = 'spherinder_mapping.png'
    save(fig, filename)


if __name__=='__main__':
    main()
    plt.show()
