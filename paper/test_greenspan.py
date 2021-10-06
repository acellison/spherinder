import os, dill
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sympy

import greenspan_inertial_waves as giw
import spherinder.operators as sph
from fileio import save_figure


g_file_prefix = 'greenspan_inertial_waves'


def makedirs():
    dirs = ['data', 'figures']
    for dir in dirs:
        path = os.path.join(dir, g_file_prefix)
        if not os.path.exists(path):
            os.makedirs(path)


def savefig(filename, *args, **kwargs):
    filename = os.path.join(os.path.join('figures', g_file_prefix), filename)
    plt.savefig(filename, *args, **kwargs)


def compute_mode(s, z, n, m, ell, normalize=False):
    mode_index = (n,(n-m)//2-ell,m)
    return giw.compute_eigenmode(s, z, *mode_index, normalize=normalize, verbose=False)


def make_domain(ns, neta, srange=(0,1), dtype='float128'):
    s, eta = np.linspace(*srange,ns, dtype=dtype), np.linspace(-1,1,neta, dtype=dtype)
    ss, ee = s[np.newaxis,:], eta[:,np.newaxis]
    zz = ee * np.sqrt(1 - ss**2)
    return s, eta, ss, zz


def analytic_mode(n, m, ell, horizontal=False):
    s, z = sympy.symbols('s z')
    simplify = sympy.simplify
   
    mode = compute_mode(s, z, n, m, ell)
    mode = simplify(mode)

    ds = lambda f: simplify(sympy.diff(f, s))
    dz = lambda f: simplify(sympy.diff(f, z))

    f = mode
    laps    = simplify(1/s * ds(s * ds(f)))
    lapphi  = simplify(-m**2/s**2 * f)
    lapz    = dz(dz(f))
    laph    = simplify(laps + lapphi)
    lapfull = simplify(laph + lapz)

    lambdify = lambda f: sympy.lambdify((s, z), f, 'numpy')
    names = ['mode', 'lap', 'laph', 'laps', 'lapphi', 'lapz']
    fns = [lambdify(fun) for fun in [mode, lapfull, laph, laps, lapphi, lapz]]
    return dict(zip(names, fns))


def plot_mode(n, m, ell):
    print('Computing analytic Laplacian')

    filename = f'data/{g_file_prefix}/greenspan_mode-n={n}-m={m}-ell={ell}.pckl'
    dill.settings['recurse'] = True
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            fns = dill.load(file)
    else:
        fns = analytic_mode(n, m, ell)
        with open(filename, 'wb') as file:
            dill.dump(fns, file)

    print('Plotting')
    srange = (0.4, 0.8)
    srange_equator = (0.95, 1.0)
    dpi = 200
    names = [('laps', 'Radial Laplacian'), ('lapphi', 'Azimuthal Laplacian'),
             ('lapz', 'Vertical Laplacian')]

    # Full Domain, Full Laplacian
    ns, neta = 256, 257
    s, eta, ss, zz = make_domain(ns, neta)

    # Mode, sphere
    mode = fns['mode'](ss, zz)
    scale = np.max(abs(mode))
    sph.plotfield(s.ravel(), eta.ravel(), mode/scale)
    plt.title('Mode')
    savefig(f'greenspan_mode_m={m}-n={n}-ell={ell}', dpi=dpi)

    # Mode, stretched coordinates
    sph.plotfield(s.ravel(), eta.ravel(), mode/scale, stretch=True)
    plt.title('Mode')
    savefig(f'greenspan_mode_m={m}-n={n}-ell={ell}-stretch', dpi=dpi)

    # Laplacian magnitudes
    def plot_profile(srange, eta):
        db = lambda x: 20*np.log10(abs(x))
        s = np.linspace(*srange, 1024, dtype='float128')
        z = np.sqrt(1-s**2) * eta
        plt.figure()
        plt.plot(s, db(fns['laps'  ](s, z)/scale), label=r'$∇_{s}^{2}$')
        plt.plot(s, db(fns['lapphi'](s, z)/scale), label=r'$∇_{\phi}^{2}$')
        plt.plot(s, db(fns['lapz'  ](s, z)/scale), label=r'$∇_{z}^{2}$')
        plt.title(f'Magnitude of Laplacian Components, η = {eta}')
        plt.xlabel('s')
        plt.ylabel('Amplitude (dB)')
        plt.legend()
        plt.grid()

    for eta in [0.0, 0.5, 1.0]:
        plot_profile((0.35, 1.0), eta=eta)
        savefig(f'greenspan_mode_m={m}-n={n}-ell={ell}-magnitudes-eta={eta}.png')

    plot_profile((0.8, 1.0), eta=0.0)
    savefig(f'greenspan_mode_m={m}-n={n}-ell={ell}-magnitudes-zoom')

    return

    # Zoomed domain: srange[0] -> srange[1]
    ns, neta = 256, 257
    s, eta, ss, zz = make_domain(ns, neta, srange=srange)

    for key, title in names:
        f = fns[key](ss, zz)
        sph.plotfield(s.ravel(), eta.ravel(), abs(f)/scale, stretch=True, aspect=None)
        plt.title(title)
        savefig(f'greenspan_mode_m={m}-n={n}-ell={ell}_{key}-axis', dpi=dpi)

    # Zoomed domain: srange_equator[0] -> srange_equator[1]
    ns, neta = 256, 257
    s, eta, ss, zz = make_domain(ns, neta, srange=(srange_equator))

    for key, title in names:
        f = fns[key](ss, zz)
        sph.plotfield(s.ravel(), eta.ravel(), abs(f)/scale, stretch=True, aspect=None)
        plt.title(title)
        savefig(f'greenspan_mode_m={m}-n={n}-ell={ell}_{key}-equator', dpi=dpi)


def filename_prefix(Lmax, Nmax, n, m, ell):
    if Lmax is None:
        LNstr = ''
    else:
        LNstr = f'-Lmax={Lmax}-Nmax={Nmax}'
    return f'figures/{g_file_prefix}/greenspan_projections-mode_m={m}-n={n}-ell={ell}{LNstr}-'


def flatten(l):
    return [item for sublist in l for item in sublist]


def project_greenspan_mode(Lmax, Nmax, n, m, ell, plot=False, Lquad=None, Nquad=None):
    def fun(t, eta):
        s = np.sqrt((1+t)/2)
        z = np.sqrt(1-s**2)*eta[:,np.newaxis]
        mode_index = (n,(n-m)//2-ell,m)
        return giw.compute_eigenmode(s, z, *mode_index, normalize=True)

    dtype = np.float128
    alpha, sigma = 0, 0
    coeffs = giw.project(fun, m, Lmax, Nmax, alpha=alpha, sigma=sigma, dtype=dtype, Lquad=Lquad, Nquad=Nquad)

    prefix = filename_prefix(Lmax, Nmax, n, m, ell)
    if plot:
        fig, ax = plt.subplots()
        for l in range(0,Lmax,2):
            ax.semilogy(abs(coeffs[l,:]), label=f'l = {l}')
        ax.set_title('Greenspan Mode Projection Magnitudes')
        ax.set_xlabel('Radial Mode Index (k)')
        ax.set_ylabel('Coefficient Magnitude')
        ax.grid(True)
        if Lmax < 24:
            ax.legend(loc='upper right')
        save_figure(prefix + 'coeffs.png', fig)

#    t, eta = np.linspace(-1,1,256,dtype=dtype), np.linspace(-1,1,257,dtype=dtype)
#    s = np.sqrt((1+t)/2)
    ns, neta = 1024, 513
    s, eta = np.linspace(0,1,ns,dtype=dtype), np.linspace(-1,1,neta,dtype=dtype)
    t = 2*s**2 - 1
    analytic_mode = fun(t, eta)
    reconstructed_mode = giw.reconstruct(coeffs, t, eta, m, Lmax, Nmax, alpha, sigma, dtype=dtype)
    error = analytic_mode - reconstructed_mode

    print(f'Lmax = {Lmax}, Nmax = {Nmax}, Lquad = {Lquad}, Nquad = {Nquad},  Max error: {np.max(abs(error))}')
    if False and plot:
        fig, ax = plt.subplots(1, 3, figsize=(13,6))
        fields = [analytic_mode, reconstructed_mode, error]
        names = ['Analytic', 'Reconstructed', 'Error']
        for i, (field, name) in enumerate(zip(fields, names)):
            cbar_format = 'log' if name == 'Error' else None
            sph.plotfield(s, eta, field, fig=fig, ax=ax[i], stretch=False, aspect=None, cbar_format=cbar_format)
            ax[i].set_title(name)
            if i > 0:
                ax[i].set_yticklabels([])
                ax[i].set_ylabel(None)

        save_figure(prefix + 'reconstruction.png', fig)

    if True and plot:
        nmodes = 4
        fig, ax = plt.subplots(2,nmodes, figsize=(nmodes*4+1,10.5))
        for i,l in enumerate(range(0,Lmax,2)):
            if i >= nmodes:
                break
            truncated = 0*coeffs
            truncated[l,:] = coeffs[l,:]
            field = giw.reconstruct(truncated, t, eta, m, Lmax, Nmax, alpha, sigma, dtype=dtype)
            sph.plotfield(s, eta, field, fig=fig, ax=ax[0][i], aspect=None, cbar_format='log', shading='auto')
            sph.plotfield(s, eta, field, fig=fig, ax=ax[1][i], aspect=None, cbar_format='log', shading='auto', stretch=True)
            ax[0][i].set_title(f'l = {l}')
            ax[0][i].set_xticklabels([])
            ax[0][i].set_xlabel(None)
            if i > 0:
                ax[0][i].set_yticklabels([])
                ax[0][i].set_ylabel(None)
                ax[1][i].set_yticklabels([])
                ax[1][i].set_ylabel(None)
            if l == 0:
                error = np.max(abs(analytic_mode - field))
                print(f'l = 0 error: {error}')
        save_figure(prefix + 'reconstruction-parts.png', fig)
        
    return np.max(abs(error))


def compute_quadrature_errors(Lmax, Nmax, n, m, ell):
    Lquad_values = np.arange(Lmax,10*Lmax+1)
    errors = 0.0 * Lquad_values
    for i,Lquad in enumerate(Lquad_values):
        errors[i] = project_greenspan_mode(Lmax, Nmax, n, m, ell, Lquad=Lquad)

    fig, ax = plt.subplots()
    ax.semilogy(Lquad_values, errors)
    ax.set_title('Error vs. Quadrature Resolution')
    ax.set_xlabel('Points in Vertical Quadrature')
    ax.set_ylabel('Max Error over Domain')
    ax.grid()
    save_figure(filename_prefix(Lmax, Nmax, n, m, ell) + 'quadrature-Lres.png', fig)

    Nquad_values = np.arange(Nmax,10*Nmax+1)
    errors = 0.0 * Nquad_values
    for i,Nquad in enumerate(Nquad_values):
        errors[i] = project_greenspan_mode(Lmax, Nmax, n, m, ell, Nquad=Nquad)

    fig, ax = plt.subplots()
    ax.semilogy(Nquad_values, errors)
    ax.set_title('Error vs. Quadrature Resolution')
    ax.set_xlabel('Points in Radial Quadrature')
    ax.set_ylabel('Max Error over Domain')
    ax.grid()
    save_figure(filename_prefix(Lmax, Nmax, n, m, ell) + 'quadrature-Nres.png', fig)


def slice_mode(n, m, ell):
    arcsins = np.linspace(0, np.pi/2, 8192)
    s = np.sin(arcsins)
    z = np.array([0.])
    mode_index = (n,(n-m)//2-ell,m)
    mode = giw.compute_eigenmode(s, z, *mode_index, normalize=True)

    fig, ax = plt.subplots()
    ax.plot(np.arcsin(s), mode)
    ax.set_xlabel(r'$\arcsin{(s)}$')
    ax.set_ylabel('Amplitude')
    ax.set_title('Mode Amplitude at $z = 0$')
    ax.grid(True)
    save_figure(filename_prefix(None, None, n, m, ell) + 'profile.png', fig)


def main():
    n, m, ell = 60, 30, 1
#    n, m, ell = 191, 95, 1

    if False:
        ns, neta = 256, 257
        s, eta, ss, zz = make_domain(ns, neta)
        mode = compute_mode(ss, zz, n, m, ell, normalize=True)
        sph.plotfield(s.ravel(), eta.ravel(), mode)

    if False:
#        compute_quadrature_errors(22, 22, 20, 10, 1)
        compute_quadrature_errors(22, 30, 60, 30, 1)

    if True:
#        project_greenspan_mode(40, 40, n, m, ell=0, plot=True)
        project_greenspan_mode(40, 40, n, m, ell=1, plot=True)
#        project_greenspan_mode(17, 32, n, m, ell, plot=True, Lquad=400, Nquad=2000)
    
    if False:
        slice_mode(n, m, ell)


if __name__=='__main__':
    makedirs()
    main()
    plt.show()

