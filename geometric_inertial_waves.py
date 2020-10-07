import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import spherinder as sph

from eigtools import eigsort
import os
import pickle
import greenspan_inertial_waves as greenspan

import config
config.internal_dtype = 'float64'


def matrices(m, Lmax, Nmax, boundary_method):
    """Construct matrices for X = [i*u(+), i*u(-), i*u(z), p]
       FIXME: artificially truncating gradient and divergence
       Notes on truncation:
        - the final ell coefficient of the u(z) equation is padded out in both
          the pressure gradient and divergence equation. 
        - the e(-) component of the pressure gradient gets truncated in n.
          we need to enforce the final n coefficient of p is zero
        - the u(+) contribution to the divergence gets truncated in n.
          we need to enforce the final n coefficient of u(+) is zero
          
    """
    ncoeff = Lmax*Nmax
    Zero = sparse.lil_matrix((ncoeff,ncoeff))
    I = sparse.eye(ncoeff)

    # Scalar gradient operators
    Gradp, Gradm, Gradz = sph.operator('grad')(m, Lmax, Nmax, alpha=0)
    Gradm = sph.resize(Gradm, Lmax, Nmax+1, Lmax, Nmax)  # truncate e(-)^* . Gradp(p)
    Gradz = sph.resize(Gradz, Lmax-1, Nmax, Lmax, Nmax)  # pad

    # Vector divergence operator
    Div = sph.operator('div')(m, Lmax, Nmax, alpha=1)
    Div = sph.resize(Div, Lmax, Nmax+1, Lmax, Nmax)      # truncate Div . e(+)^* . u
    Divp = Div[:,:ncoeff]
    Divm = Div[:,ncoeff:2*ncoeff]
    Divz = Div[:,2*ncoeff:]

    # Boundary condition
    Rad = sph.operator('erdot')(m, Lmax, Nmax, alpha=1)
    Boundary = sph.operator('boundary')(m, Lmax+1, Nmax+1, alpha=1, sigma=0)
    Bound = Boundary @ Rad
    Bound = sph.remove_zero_rows(Bound)

    # Tau conversion
    # Time derivative matrices
    M00 = I
    M11 = I
    M22 = I
    M33 = Zero

    # i*u+ equation - spin+ velocity component
    L00 = -2 * I
    L01 = Zero
    L02 = Zero
    L03 = -Gradp

    # i*u- equation - spin- velocity component
    L10 = Zero
    L11 = 2 * I
    L12 = Zero
    L13 = -Gradm

    # i*w equation - vertical velocity component
    L20 = Zero
    L21 = Zero
    L22 = Zero
    L23 = -Gradz

    # Divergence equation
    L30 = Divp
    L31 = Divm
    L32 = Divz
    L33 = Zero

    Mmats = [M00, M11, M22, M33]
    upmats = [L00, L01, L02, L03]
    ummats = [L10, L11, L12, L13]
    uzmats = [L20, L21, L22, L23]
    pmats = [L30, L31, L32, L33]

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([upmats, ummats, uzmats, pmats], format=sparse_format)

    # Boundary conditions
    def impenetrable():
        ntau = np.shape(Bound)[0]
        row = sparse.hstack([Bound, sparse.lil_matrix((ntau,ncoeff))])

        # Tau contribution in the final ell coefficients
        alpha_bc = 1
        Convz1 = sph.convert_alpha_up_n(1, m, Lmax, Nmax, alpha=0, sigma=0, truncate=True)
        Convz2 = sph.convert_alpha_up_n(2-alpha_bc, m, Lmax, Nmax, alpha=alpha_bc, sigma=0, truncate=True)
        conv1, conv2 = Convz1[:,-Nmax:], Convz2[:,-Nmax:]
        col1 = sparse.bmat([[0*conv1,0*conv2],
                            [0*conv1,0*conv2],
                            [  conv1,0*conv2],
                            [0*conv1,  conv2]])

        # Tau contribution in final radial coefficient
        whichtau = (0,)
        Convp = sph.convert_alpha_up_n(1, m, Lmax, Nmax, alpha=0, sigma=+1, truncate=True)
        col2 = Convp[:,Nmax-1::Nmax]
        col2 = sparse.bmat([[(whichtau[j]==i)*col2 for j in range(len(whichtau))] for i in range(4)])

        col = sparse.hstack([col1,col2])

        return row, col

    # Create the boundary condition rows and tau columns
    row, col = impenetrable()
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
    prefix = 'geometric_inertial_waves'
    return os.path.join(basepath, os.path.join(prefix, prefix))


def pickle_filename(m, Lmax, Nmax, boundary_method, directory='data'):
    return filename_prefix(directory) + '-evalues-m={}-Lmax={}-Nmax={}-{}.pckl'.format(m,Lmax,Nmax,boundary_method)


def solve_eigenproblem(m, Lmax, Nmax, boundary_method):
    # Construct the system
    M, L = matrices(m, Lmax, Nmax, boundary_method)

    plot_spy = True
    if plot_spy:
        fig, plot_axes = plt.subplots(1,2,figsize=(9,4))
        plot_axes[0].spy(L)
        plot_axes[1].spy(M)
        plot_axes[0].set_title('L')
        plot_axes[1].set_title('M')
        filename = filename_prefix('figures') + f'-m={m}-Lmax={Lmax}-Nmax={Nmax}-spy.png'
        savefig(filename)
        plt.show()

    # Compute the eigenvalues and eigenvectors
    print('Eigenvalue problem, size {}'.format(np.shape(L)))
    evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)

    # Output data
    data = {'m': m, 'Lmax': Lmax, 'Nmax': Nmax, 
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors}
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    savedata(filename, data)


def expand_evectors(m, Lmax, Nmax, boundary_method, vec, s, eta):
    z = 2*s**2 - 1

    ncoeff = Lmax*Nmax
    tau = vec[4*ncoeff:]

    upbasis = [sph.psi(Nmax, m, ell, s, eta, sigma=+1, alpha=1) for ell in range(Lmax)]
    umbasis = [sph.psi(Nmax, m, ell, s, eta, sigma=-1, alpha=1) for ell in range(Lmax)]
    uzbasis = [sph.psi(Nmax, m, ell, s, eta, sigma= 0, alpha=1) for ell in range(Lmax)]
    pbasis  = [sph.psi(Nmax, m, ell, s, eta, sigma= 0, alpha=0) for ell in range(Lmax)]

    # Get the grid space vector fields
    vec = vec.astype(np.complex128)
    vec[:3*ncoeff] /= 1j
    upcoeff = vec[:ncoeff] 
    umcoeff = vec[ncoeff:2*ncoeff] 
    uzcoeff = vec[2*ncoeff:3*ncoeff]
    pcoeff = vec[3*ncoeff:4*ncoeff]

    # Convert to grid space
    up = sph.expand(upbasis, np.reshape(upcoeff, (Lmax,Nmax)))
    um = sph.expand(umbasis, np.reshape(umcoeff, (Lmax,Nmax)))
    uz = sph.expand(uzbasis, np.reshape(uzcoeff, (Lmax,Nmax)))
    p  = sph.expand( pbasis, np.reshape( pcoeff, (Lmax,Nmax)))
    u, v, w = np.sqrt(0.5)*(up + um), -1j*np.sqrt(0.5)*(up - um), uz

    ns, neta = len(s), len(eta)
    ss, ee = s.reshape(1,ns), eta.reshape(neta,1)
    ur = ss * u + ee * np.sqrt(1-ss**2) * w 

    return u, v, w, p, tau, ur


def plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_fields):
    save_plots = False
    plot_field_indices = [3]
#    plot_field_indices = list(range(4))
    n, ell = 150, 6
    mode_index = (n,(n-m)//2+ell,m)

    # Load the data
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    if save_plots:
        def save(fn): savefig(fn)
    else:
        def save(_): pass

    evalue_target = 2*greenspan.compute_eigenvalues(mode_index[0], mode_index[2])[mode_index[1]-1]
    modestr = str(mode_index[0]) + str(mode_index[1]) + str(mode_index[2])
    configstr = 'm={}-Lmax={}-Nmax={}-{}'.format(m,Lmax,Nmax,boundary_method)
    prefix = filename_prefix('figures')

    nlarge = len(np.where(np.abs(evalues.real) > 2)[0])
    evalues_in_range = evalues[np.where(np.abs(evalues.real) <= 2)]
    nimag = len(np.where(np.abs(evalues_in_range.imag) > 1e-15)[0])
    maximag = np.max(np.abs(evalues_in_range.imag))
    print('Number of bad eigenvalues: {}/{}'.format(nimag+nlarge,len(evalues)))
    print('    - Number out of real interval [-2,2]: {}/{}'.format(nlarge,len(evalues)))
    if nlarge > 0:
        maxind = np.argmax(np.abs(evalues.real))
        print('        - Maximum real part: {}'.format(evalues[maxind].real))
    print('    - Number with nonzero imaginary part: {}/{}'.format(nimag,len(evalues)))
    if nimag > 0:
        print('        - Maximum imaginary part: {}'.format(maximag))

    # Plot the eigenvalues
    if plot_evalues:
        fig, ax = plt.subplots()
        ax.plot(evalues_in_range.real, evalues_in_range.imag, '.', markersize=3, color='tab:blue')
        ax.grid()
        plt.xlim([-2.1,2.1])
        ax.set_xlabel('Real(λ)')
        ax.set_ylabel('Imag(λ)')
        ax.set_title('Inertial Wave Eigenvalues in the Stretched Sphere')
        fig.set_tight_layout(True)

        filename = prefix + '-evalues-' + configstr + '.png'
        save(filename)

    if not plot_fields:
        return

    # Get the target eigenpair
    index = np.argmin(abs(evalues - evalue_target))
    val, vec = evalues[index], evectors[:,index]

    print('Plotting eigenvector with eigenvalue {:1.4f}'.format(val))

    # Construct the basis polynomials
    ns, neta = 256, 255
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    u, v, w, p, tau, ur = expand_evectors(m, Lmax, Nmax, boundary_method, vec, s, eta)

    # Spherical radial velocity component
    error_top = np.max(np.abs(ur[-1,:]))
    error_bot = np.max(np.abs(ur[ 0,:]))
    print('Top    boundary error: {:1.3e}'.format(error_top))
    print('Bottom boundary error: {:1.3e}'.format(error_bot))

    fields = [u,v,w,p,ur]
    field_names = ['u','v','w','p','u_r']

    for i in range(len(plot_field_indices)):
        field_index = plot_field_indices[i]
        Fgrid = fields[field_index]
        relative_real = np.linalg.norm(np.real(Fgrid))/np.linalg.norm(Fgrid)
        f = Fgrid.real if relative_real > 0.5 else Fgrid.imag

        sph.plotfield(s, eta, f)
        plt.title(r'${}$'.format(field_names[field_index]))
        filename = prefix + '-evector-' + configstr + '-' + modestr + '-' + field_names[field_index] + '.png'
        save(filename)


def main():
    solve = True
    plot_evalues = True
    plot_fields = True

    m = 95
    Lmax, Nmax = 8, 8
    boundary_method = 'tau'

    print('Inertial Waves, m = {}'.format(m))
    print('  Domain size: Lmax = {}, Nmax = {}'.format(Lmax, Nmax))

    if solve:
        solve_eigenproblem(m, Lmax, Nmax, boundary_method)

    if plot_fields or plot_evalues:
        plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_fields)
        plt.show()


if __name__=='__main__':
    main()
