import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from dedalus_sphere import jacobi as Jacobi
from eigtools import eigsort
import os
import pickle
import greenspan_inertial_waves as greenspan

import spherinder as sph


def matrices(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh):
    """Construct matrices for X = [i*u(+), i*u(-), i*w, p]
    """
    ncoeff = Lmax*Nmax
    Zero = sparse.lil_matrix((ncoeff,ncoeff))
    I = sparse.eye(ncoeff)

    # Scalar laplacian
    Laps = sph.operator('lap')(m, Lmax, Nmax, alpha=0)
    Laps = sph.resize(Laps, Lmax, Nmax+1, Lmax, Nmax)

    # Vector laplacian
    Lapp, Lapm, Lapz = sph.operator('lap', 'vector')(m, Lmax, Nmax, alpha=1)
    Lapp = sph.resize(Lapp, Lmax, Nmax+1, Lmax, Nmax)
    Lapm = sph.resize(Lapm, Lmax, Nmax+1, Lmax, Nmax)
    Lapz = sph.resize(Lapz, Lmax, Nmax+1, Lmax, Nmax)

    # Vector divergence operator
    Div = sph.operator('div')(m, Lmax, Nmax, alpha=1)
    Div = sph.resize(Div, Lmax, Nmax+1, Lmax, Nmax)      # truncate Div . e(+)^* . u
    Divp, Divm, Divz = Div[:,:ncoeff], Div[:,ncoeff:2*ncoeff], Div[:,2*ncoeff:]

    # Convert operators to the proper alpha spaces
    # u(+) conversion from alpha=1 to alpha=3
    Cp = sph.convert_alpha_up_n(2, m, Lmax, Nmax, alpha=1, sigma=+1, truncate=True)

    # u(-) conversion from alpha=1 to alpha=3
    Cm = sph.convert_alpha_up_n(2, m, Lmax, Nmax, alpha=1, sigma=-1, truncate=True)

    # u(z) conversion from alpha=1 to alpha=3
    Cz = sph.convert_alpha_up_n(2, m, Lmax, Nmax, alpha=1, sigma=0, truncate=True)
 
    # scalar conversion from alpha=0 to alpha=2
    Cs = sph.convert_alpha_up_n(2, m, Lmax, Nmax, alpha=0, sigma=0, truncate=True)

    # Pressure gradient: convert from alpha=1 to alpha=3
    Gradp, Gradm, Gradz = sph.operator('grad')(m, Lmax, Nmax, alpha=0)
    Cgp = sph.convert_alpha_up_n(2, m, Lmax,   Nmax,   alpha=1, sigma=+1, truncate=False)
    Cgm = sph.convert_alpha_up_n(2, m, Lmax,   Nmax+1, alpha=1, sigma=-1, truncate=False)
    Cgz = sph.convert_alpha_up_n(2, m, Lmax-1, Nmax,   alpha=1, sigma=0,  truncate=False)
    Gradp, Gradm, Gradz = Cgp @ Gradp, Cgm @ Gradm, Cgz @ Gradz
    Gradp = sph.resize(Gradp, Lmax,   Nmax+2, Lmax, Nmax)
    Gradm = sph.resize(Gradm, Lmax,   Nmax+3, Lmax, Nmax)
    Gradz = sph.resize(Gradz, Lmax-1, Nmax+2, Lmax, Nmax)

    # Radial vector extraction, convert from alpha=1 to alpha=2
    Rad = sph.operator('erdot')(m, Lmax, Nmax, alpha=1)
    Cr = sph.operator('conversion')(m, Lmax+1, Nmax+1, alpha=1, sigma=0)
    Rad = Cr @ Rad    
    Rad = sph.resize(Rad, Lmax+1, Nmax+2, Lmax, Nmax)
    Radp, Radm, Radz = Rad[:,:ncoeff], Rad[:,ncoeff:2*ncoeff], Rad[:,2*ncoeff:]

    # Radial vector multiplication r e_r * T, convert from alpha=1 to alpha=3
    RTp, RTm, RTz = sph.operator('rtimes')(m, Lmax, Nmax, alpha=0)
    CrTp = sph.convert_alpha_up_n(2, m, Lmax,   Nmax+1,   alpha=1, sigma=+1, truncate=False)
    CrTm = sph.convert_alpha_up_n(2, m, Lmax,   Nmax+2, alpha=1, sigma=-1, truncate=False)
    CrTz = sph.convert_alpha_up_n(2, m, Lmax+1, Nmax+2, alpha=1, sigma=0,  truncate=False)
    RTp, RTm, RTz = CrTp @ RTp, CrTm @ RTm, CrTz @ RTz
    RTp = sph.resize(RTp, Lmax,   Nmax+3, Lmax, Nmax)
    RTm = sph.resize(RTm, Lmax,   Nmax+4, Lmax, Nmax)
    RTz = sph.resize(RTz, Lmax+1, Nmax+4, Lmax, Nmax)

    # Boundary operator
    Boundary = sph.operator('boundary')
    Boundp = Boundary(m, Lmax, Nmax, alpha=1, sigma=+1)
    Boundm = Boundary(m, Lmax, Nmax, alpha=1, sigma=-1)
    Boundz = Boundary(m, Lmax, Nmax, alpha=1, sigma=0)
    BoundT = Boundary(m, Lmax, Nmax, alpha=0, sigma=0)
    Bound = sparse.bmat([[  Boundp, 0*Boundm, 0*Boundz, 0*BoundT, 0*BoundT],
                         [0*Boundp,   Boundm, 0*Boundz, 0*BoundT, 0*BoundT],
                         [0*Boundp, 0*Boundm,   Boundz, 0*BoundT, 0*BoundT],
                         [0*Boundp, 0*Boundm, 0*Boundz, 0*BoundT,   BoundT]])

    # Time derivative matrices
    M00 = Ekman * Cp
    M11 = Ekman * Cm
    M22 = Ekman * Cz
    M33 = Zero
    M44 = Prandtl * Cs

    # i*u+ equation - spin+ velocity component
    L00 = -1j * Cp + Ekman * Lapp
    L01 = Zero
    L02 = Zero
    L03 = -Gradp
    L04 = Rayleigh * RTp

    # i*u- equation - spin- velocity component
    L10 = Zero
    L11 = 1j * Cm + Ekman * Lapm
    L12 = Zero
    L13 = -Gradm
    L14 = Rayleigh * RTm

    # i*w equation - vertical velocity component
    L20 = Zero
    L21 = Zero
    L22 = Ekman * Lapz
    L23 = -Gradz
    L24 = Rayleigh * RTz

    # Divergence equation
    L30 = Divp
    L31 = Divm
    L32 = Divz
    L33 = Zero
    L34 = Zero

    # Temperature equation
    L40 = Prandtl * Radp
    L41 = Prandtl * Radm
    L42 = Prandtl * Radz
    L43 = Zero
    L44 = Laps

    Mmats = [M00, M11, M22, M33, M44]
    upmats = [L00, L01, L02, L03, L04]
    ummats = [L10, L11, L12, L13, L14]
    uzmats = [L20, L21, L22, L23, L24]
    pmats = [L30, L31, L32, L33, L34]
    Tmats = [L40, L41, L42, L43, L44]

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([upmats, ummats, uzmats, pmats, Tmats], format=sparse_format)

    # Boundary conditions
    def no_slip():
        row = Bound

        alpha_bc, alpha_bc_T = 0, 0
        Taup = sph.convert_alpha_up_n(3-alpha_bc, m, Lmax, Nmax, alpha=alpha_bc, sigma=+1, truncate=True)
        Taum = sph.convert_alpha_up_n(3-alpha_bc, m, Lmax, Nmax, alpha=alpha_bc, sigma=-1, truncate=True)
        Tauz = sph.convert_alpha_up_n(3-alpha_bc, m, Lmax, Nmax, alpha=alpha_bc, sigma=0, truncate=True)
        TauT = sph.convert_alpha_up_n(2-alpha_bc_T, m, Lmax, Nmax, alpha=alpha_bc_T, sigma=0, truncate=True)
        taup, taum, tauz, tauT = Taup[:,-2*Nmax:], Taum[:,-2*Nmax:], Tauz[:,-2*Nmax:], TauT[:,-2*Nmax:]

        col1 = sparse.bmat([[  taup,0*taum,0*tauz,0*tauT],
                            [0*taup,  taum,0*tauz,0*tauT],
                            [0*taup,0*taum,  tauz,0*tauT],
                            [0*taup,0*taum,0*tauz,0*tauT],
                            [0*taup,0*taum,0*tauz,  tauT]])

        taup, taum, tauz, tauT = Taup[:,Nmax-1:-2*Nmax:Nmax], Taum[:,Nmax-1:-2*Nmax:Nmax], \
                                 Tauz[:,Nmax-1:-2*Nmax:Nmax], TauT[:,Nmax-1:-2*Nmax:Nmax]
        col2 = sparse.bmat([[  taup,0*taum,0*tauz,0*tauT],
                            [0*taup,  taum,0*tauz,0*tauT],
                            [0*taup,0*taum,  tauz,0*tauT],
                            [0*taup,0*taum,0*tauz,0*tauT],
                            [0*taup,0*taum,0*tauz,  tauT]])
 
        col = sparse.hstack([col1,col2])
        return row, col

    # Create the boundary condition rows and tau columns
    row, col = no_slip()

    corner = np.zeros((np.shape(row)[0], np.shape(col)[1]))
    L = sparse.bmat([[  L, col],
                     [row, corner]], format='csr')

    M = sparse.bmat([[    M, 0*col],
                     [0*row, corner]], format='csr')

    M, L = M.tocsr(), L.tocsr()
    return M, L


def checkdir(filename):
    filename = os.path.abspath(filename)
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.mkdir(path)

def savedata(filename, data):
    checkdir(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def savefig(filename):
    checkdir(filename)
    plt.savefig(filename)
    

def filename_prefix(directory='data'):
    basepath = os.path.join(os.path.dirname(__file__), directory)
    prefix = 'geometric_linear_onset'
    return os.path.join(basepath, os.path.join(prefix, prefix))


def pickle_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, directory='data'):
    return filename_prefix(directory) + f'-evalues-m={m}-Lmax={Lmax}-Nmax={Nmax}-Ekman={Ekman:1.4e}-Prandtl={Prandtl}-Rayleigh={Rayleigh:1.4e}-{boundary_method}.pckl'


def solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh):
    # Construct the system
    M, L = matrices(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh)

    plot_spy = False
    if plot_spy:
        fig, plot_axes = plt.subplots(1,2,figsize=(9,4))
        plot_axes[0].spy(L)
        plot_axes[1].spy(M)
        plt.show()

    # Compute the eigenvalues and eigenvectors
    print('Eigenvalue problem, size {}'.format(np.shape(L)))
    evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)

    # Output data
    data = {'m': m, 'Lmax': Lmax, 'Nmax': Nmax, 
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors,
            'Ekman': Ekman, 'Prandtl': Prandtl, 'Rayleigh': Rayleigh}
    filename = pickle_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh)
    savedata(filename, data)


def expand_evectors(m, Lmax, Nmax, boundary_method, vec, s, eta):
    z = 2*s**2 - 1

    ncoeff = Lmax*Nmax

    upbasis = [sph.psi(Nmax, m, ell, s, eta, sigma=+1, alpha=1) for ell in range(Lmax)]
    umbasis = [sph.psi(Nmax, m, ell, s, eta, sigma=-1, alpha=1) for ell in range(Lmax)]
    uzbasis = [sph.psi(Nmax, m, ell, s, eta, sigma= 0, alpha=1) for ell in range(Lmax)]
    pbasis  = [sph.psi(Nmax, m, ell, s, eta, sigma= 0, alpha=0) for ell in range(Lmax)]
    Tbasis  = [sph.psi(Nmax, m, ell, s, eta, sigma= 0, alpha=0) for ell in range(Lmax)]

    # Get the grid space vector fields
    vec = vec.astype(np.complex128)
    upcoeff = vec[:ncoeff] 
    umcoeff = vec[ncoeff:2*ncoeff] 
    uzcoeff = vec[2*ncoeff:3*ncoeff]
    pcoeff = vec[3*ncoeff:4*ncoeff]
    Tcoeff = vec[4*ncoeff:5*ncoeff]
    tau = vec[5*ncoeff:]

    # Convert to grid space
    up = sph.expand(upbasis, np.reshape(upcoeff, (Lmax,Nmax)))
    um = sph.expand(umbasis, np.reshape(umcoeff, (Lmax,Nmax)))
    uz = sph.expand(uzbasis, np.reshape(uzcoeff, (Lmax,Nmax)))
    p  = sph.expand( pbasis, np.reshape( pcoeff, (Lmax,Nmax)))
    T  = sph.expand( Tbasis, np.reshape( Tcoeff, (Lmax,Nmax)))
    u, v, w = np.sqrt(0.5)*(up + um), -1j * np.sqrt(0.5)*(up - um), uz

    return u, v, w, p, T, tau


def plot_solution(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, omega, plot_evalues, plot_fields):
    save_plots = True
    plot_field_indices = [0,2,4]

    # Load the data
    filename = pickle_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    if save_plots:
        def save(fn): savefig(fn)
    else:
        def save(_): pass

    evalue_target = omega
    configstr = f'-m={m}-Lmax={Lmax}-Nmax={Nmax}-Ekman={Ekman:1.4e}-Prandtl={Prandtl}-Rayleigh={Rayleigh:1.4e}-{boundary_method}'
    prefix = filename_prefix('figures')

    # Plot the eigenvalues
    if plot_evalues:
        fig, ax = plt.subplots()
        ax.plot(evalues.real, evalues.imag, '.')
        ax.grid()
        ax.set_xlabel('Real(λ)')
        ax.set_ylabel('Imag(λ)')
        ax.set_title('Linear Onset Eigenvalues in the Stretched Sphere')
        fig.set_tight_layout(True)

        filename = prefix + configstr + '-evalues.png'
        save(filename)

    if not plot_fields:
        return

    # Get the target eigenpair
    index = np.argmin(abs(evalues - evalue_target))
    val, vec = evalues[index], evectors[:,index]

    print('Omega target: {:1.4f}'.format(omega))
    print('Plotting eigenvector with eigenvalue {:1.4f}'.format(val))

    # Construct the basis polynomials
    ns, neta = 256, 255
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    u, v, w, p, T, tau = expand_evectors(m, Lmax, Nmax, boundary_method, vec, s, eta)

    fields = [u,v,w,p,T]
    field_names = ['u','v','w','p','T']

    for i in range(len(plot_field_indices)):
        field_index = plot_field_indices[i]
        Fgrid = fields[field_index]
        relative_real = np.linalg.norm(np.real(Fgrid))/np.linalg.norm(Fgrid)
        f = Fgrid.real if relative_real > 0.5 else Fgrid.imag

        sph.plotfield(s, eta, f)
        plt.title(r'${}$'.format(field_names[field_index]))
        filename = prefix + configstr + '-evector-' + field_names[field_index] + '.png'
        save(filename)


def rotation_configs():
    return [{'Ekman': 10**-4,   'm': 6,  'omega': -.43346, 'Rayleigh': 5.1549, 'Lmax': 16, 'Nmax': 16},
            {'Ekman': 10**-4.5, 'm': 9,  'omega': -.44276, 'Rayleigh': 4.7613, 'Lmax': 16, 'Nmax': 16},
            {'Ekman': 10**-5,   'm': 14, 'omega': -.45715, 'Rayleigh': 4.5351, 'Lmax': 16, 'Nmax': 24},
            {'Ekman': 10**-5.5, 'm': 20, 'omega': -.45760, 'Rayleigh': 4.3937, 'Lmax': 16, 'Nmax': 24},
            {'Ekman': 10**-6,   'm': 30, 'omega': -.46394, 'Rayleigh': 4.3021, 'Lmax': 16, 'Nmax': 24},
            {'Ekman': 10**-6.5, 'm': 44, 'omega': -.46574, 'Rayleigh': 4.2416, 'Lmax': 16, 'Nmax': 32},
            {'Ekman': 10**-7,   'm': 65, 'omega': -.46803, 'Rayleigh': 4.2012, 'Lmax': 16, 'Nmax': 32},
            {'Ekman': 10**-7.5, 'm': 95, 'omega': -.46828, 'Rayleigh': 4.1742, 'Lmax': 16, 'Nmax': 32}]

def main():
    solve = False
    plot_evalues = True
    plot_fields = True

    config_index = 2
    config = rotation_configs()[config_index]

    m, Ekman, Prandtl, Rayleigh, omega = config['m'], config['Ekman'], 1, config['Rayleigh'], config['omega']
    Lmax, Nmax = config['Lmax'], config['Nmax']
    boundary_method = 'tau'

    omega = 1j*omega/Ekman**(2/3)
    Rayleigh /= Ekman**(1/3)

    print(f'Linear onset, m = {m}, Ekman = {Ekman:1.4e}, Prandtl = {Prandtl}, Rayleigh = {Rayleigh:1.4e}')
    print('  Domain size: Lmax = {}, Nmax = {}'.format(Lmax, Nmax))

    if solve:
        solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh)

    if plot_fields or plot_evalues:
        plot_solution(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, omega, plot_evalues, plot_fields)
        plt.show()


if __name__=='__main__':
    main()
