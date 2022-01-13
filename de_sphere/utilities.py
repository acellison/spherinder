import os, pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
from dedalus_sphere import ball128, sphere128

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_ball(m, L_max, N_max):
    """Don't actually construct the ball, just hold the configuration in an attribute dictionary.
       Saves memory while performing the eigensolve"""
    R_max = 3
    m_start, m_end = m, m
    ell_start, ell_end = m, L_max

    B = AttrDict({'N_max':N_max, 'L_max':L_max, 'R_max':R_max, 'ell_min':ell_start, 'ell_max':ell_end, 'm_min':m_start, 'm_max':m_end, 'a':0.})

    # Make domain
    domain = None

    return B, domain


def make_tensor_coeffs(m, L_max, N_max, R_max, rank, dtype='float64'):
    coeffs = []
    for ell in range(m, L_max+1):
        N = N_max+1 - ball128.N_min(ell-R_max)
        coeffs.append(np.zeros((3**rank*N,1), dtype=dtype))
    return coeffs


def expand_field(field, m, L_max, N_max, R_max, z, cos_theta):
    """Expand a field.  For now only supports scalar fields"""
    a = 0      # Default offset for Jacobi polynomials
    s = 0      # Spin weight
    alpha = 0  # How many times we've been differentiated
    R_max = 3  # Max rank

    Y = sphere128.Y(L_max, m, s, cos_theta).T

    f = 0*z[np.newaxis,:] + 0*cos_theta[:,np.newaxis]
    for ell in range(m, L_max+1):
        ell_local = ell-m
        N = N_max - ball128.N_min(ell-R_max)

        Qn = ball128.polynomial(N,alpha,ell,z,a=a)
        frad = field[ell_local].T @ Qn
        Yell = Y[:,ell_local][:,np.newaxis]
        f = f + frad * Yell

    return f


def plot_fields(fielddict, z, cos_theta, colorbar=True, fig=None, ax=None, cmap=None):
    shading = 'gouraud'  # 'gouraud' interpolates but is slower than 'nearest'
    r = np.sqrt((z+1)/2)
    r, cos_theta = r[np.newaxis,:], cos_theta[:,np.newaxis]
    sin_theta = np.sqrt(1-cos_theta**2)
    x, z = r*sin_theta, r*cos_theta
    for name, field in fielddict.items():
        if fig is None or ax is None:
            fig, ax = plt.subplots(1,1,figsize=(4.25,6))

        if cmap is None:
            cmap = plt.get_cmap('RdBu')
        cmap = copy.copy(cmap)
        cmap.set_bad(color='grey', alpha=.5)
        if shading == 'gouraud':
            # Shade via interpolation.  Can handle non-monotonic input grids
            c_im = ax.pcolormesh(x, z, field, cmap=cmap, shading='gouraud')
        else:
            # Nearest shading requires a monotonic input grid
            n = len(cos_theta)
            _    = ax.pcolormesh(x[:n,:], z[:n,:], field[:n,:], cmap=cmap, shading='nearest')
            c_im = ax.pcolormesh(x[n:,:], z[n:,:], field[n:,:], cmap=cmap, shading='nearest')
        if colorbar:
            cbar = fig.colorbar(c_im)
        eps = 0.02
        ax.plot((1+eps/2)*sin_theta, (1+eps/2)*cos_theta, color='k', linewidth=1)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(name)
        ax.set_aspect('equal')

        fig.show()


def checkdir(filename):
    path = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(path):
        os.makedirs(path)


def save_data(filename, data):
    checkdir(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def save_figure(filename, fig):
    checkdir(filename)
    fig.set_tight_layout(True)
    fig.savefig(filename)


