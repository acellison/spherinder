import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import os
import pickle

import greenspan_inertial_waves as greenspan

internal_dtype = 'float128'
from spherinder import config
config.internal_dtype = internal_dtype

import spherinder.spherinder as sph
from spherinder.eigtools import eigsort
from fileio import save_data, save_figure


g_file_prefix = 'spherinder_inertial_waves'

triangular_truncation = False


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
    alpha_bc_p = 0    # tau polynomial basis in u(+) equation, no greater than 1
    alpha_bc_z = 0    # tau polynomial basis in u(z) equation, no greater than 1
    alpha_bc_div = 0  # tau polynomial basis in div equation, no greater than 2
    beta_bc_p = 0     # tau polynomial basis shift, no greater than 1
    beta_bc_z = 0     # tau polynomial basis shift, no greater than 1
    beta_bc_div = 0   # tau polynomial basis shift, no greater than 2

    if alpha_bc_p > 1:
        raise ValueError('Tau polynomial has alpha too high in p equation')
    if alpha_bc_z > 1:
        raise ValueError('Tau polynomial has alpha too high in z equation')
    if alpha_bc_div > 2:
        raise ValueError('Tau polynomial has alpha too high in div equation')
    if beta_bc_p > 1:
        raise ValueError('Tau polynomial has beta too high in p equation')
    if beta_bc_z > 1:
        raise ValueError('Tau polynomial has beta too high in z equation')
    if beta_bc_div > 2:
        raise ValueError('Tau polynomial has beta too high in div equation')


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
    Rad = sph.operator('erdot', dtype='float128')(m, Lmax, Nmax, alpha=1)
    Boundary = sph.operator('boundary', dtype='float128', internal='float128')(m, Lmax+1, Nmax+1, alpha=1, sigma=0)
    Bound = Boundary @ Rad
    Bound = sph.remove_zero_rows(Bound).astype('float64')

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

    if triangular_truncation:
        Mmats = [sph.triangular_truncate(mat, Lmax, Nmax) for mat in Mmats]
        upmats = [sph.triangular_truncate(mat, Lmax, Nmax) for mat in upmats]
        ummats = [sph.triangular_truncate(mat, Lmax, Nmax) for mat in ummats]
        uzmats = [sph.triangular_truncate(mat, Lmax, Nmax) for mat in uzmats]
        pmats = [sph.triangular_truncate(mat, Lmax, Nmax) for mat in pmats]

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([upmats, ummats, uzmats, pmats], format=sparse_format)

    # Boundary conditions
    def impenetrable():
        ntau = np.shape(Bound)[0]
        row = sparse.hstack([Bound, sparse.lil_matrix((ntau,ncoeff))])

        # Tau contribution in the final ell coefficients
        if beta_bc_z > 0:
            Convz1 = sph.convert_beta(m, Lmax, Nmax, alpha=1, sigma=0, beta=beta_bc_z)
        else:
            Convz1 = sph.convert_alpha(1-alpha_bc_z, m, Lmax, Nmax, alpha=alpha_bc_z, sigma=0, truncate=True)

        if beta_bc_div > 0:
            Convz2 = sph.convert_beta(m, Lmax, Nmax, alpha=2, sigma=0, beta=beta_bc_div)
        else:
            Convz2 = sph.convert_alpha(2-alpha_bc_div, m, Lmax, Nmax, alpha=alpha_bc_div, sigma=0, truncate=True)

        conv1, conv2 = Convz1[:,-Nmax:], Convz2[:,-Nmax:]
        col1 = sparse.bmat([[0*conv1,0*conv2],
                            [0*conv1,0*conv2],
                            [  conv1,0*conv2],
                            [0*conv1,  conv2]])

        # Tau contribution in final radial coefficient
        whichtau = (0,)
        if beta_bc_p > 0:
            Convp = sph.convert_beta(m, Lmax, Nmax, alpha=1, sigma=+1, beta=beta_bc_p)
        else:
            Convp = sph.convert_alpha(1-alpha_bc_p, m, Lmax, Nmax, alpha=alpha_bc_p, sigma=+1, truncate=True)
        col2 = Convp[:,Nmax-1::Nmax]
        col2 = sparse.vstack([col2,0*col2,0*col2,0*col2])

        col = sparse.hstack([col1,col2])
        return row, col

    # Create the boundary condition rows and tau columns
    row, col = impenetrable()
    if triangular_truncation:
        Nout = lambda ell: Nmax-ell
        row, col = row.tocsr(), col.tocsr()
        row = sparse.hstack([sph.resize(row[:,i*ncoeff:(i+1)*ncoeff].T, Lmax, Nmax, Lmax, Nout).T for i in range(4)])
        col = sparse.vstack([sph.resize(col[i*ncoeff:(i+1)*ncoeff,:],   Lmax, Nmax, Lmax, Nout) for i in range(4)])

    corner = np.zeros((np.shape(row)[0], np.shape(col)[1]))
    L = sparse.bmat([[  L, col],
                     [row, corner]], format='csr')

    M = sparse.bmat([[    M, 0*col],
                     [0*row, corner]], format='csr')

    M, L = M.tocsr(), L.tocsr()
    return M, L


def filename_prefix(directory='data'):
    basepath = os.path.join(os.path.dirname(__file__), directory)
    prefix = g_file_prefix
    return os.path.join(basepath, os.path.join(prefix, prefix))


def pickle_filename(m, Lmax, Nmax, boundary_method, directory='data'):
    return filename_prefix(directory) + '-evalues-m={}-Lmax={}-Nmax={}-{}.pckl'.format(m,Lmax,Nmax,boundary_method)


def solve_eigenproblem(m, Lmax, Nmax, boundary_method):
    # Construct the system
    M, L = matrices(m, Lmax, Nmax, boundary_method)

    plot_spy = False
    if plot_spy:
        fig, plot_axes = plt.subplots(1,2,figsize=(9,4))
        plot_axes[0].spy(L)
        plot_axes[1].spy(M)
        plot_axes[0].set_title('L')
        plot_axes[1].set_title('M')
        filename = filename_prefix('figures') + f'-m={m}-Lmax={Lmax}-Nmax={Nmax}-spy.png'
        save_figure(filename, fig)
        plt.show()

    # Compute the eigenvalues and eigenvectors
    print('Eigenvalue problem, size {}'.format(np.shape(L)))
    evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)

    # Output data
    data = {'m': m, 'Lmax': Lmax, 'Nmax': Nmax, 
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors}
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    save_data(filename, data)


def expand_evectors(m, Lmax, Nmax, boundary_method, vec, s, eta):
    z = 2*s**2 - 1

    if triangular_truncation:
        Nout = lambda ell: Nmax-ell
        ncoeff = np.sum([Nout(ell) for ell in range(Lmax)])
    else:
        ncoeff = Lmax*Nmax
    tau = vec[4*ncoeff:]

    galerkin = boundary_method == 'galerkin'
    upbasis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=+1, alpha=1, galerkin=galerkin)
    umbasis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=-1, alpha=1, galerkin=galerkin)
    uzbasis = sph.Basis(s, eta, m, Lmax, Nmax, sigma=0,  alpha=1, galerkin=galerkin)
    pbasis  = sph.Basis(s, eta, m, Lmax, Nmax, sigma=0,  alpha=0, galerkin=False)

    # Get the grid space vector fields
    vec = vec.astype(np.complex128)
    vec[:3*ncoeff] /= 1j
    upcoeff = vec[:ncoeff] 
    umcoeff = vec[ncoeff:2*ncoeff] 
    uzcoeff = vec[2*ncoeff:3*ncoeff]
    pcoeff = vec[3*ncoeff:4*ncoeff]

    # Convert to grid space
    if triangular_truncation:
        bases = {'up':upbasis, 'um':umbasis, 'uz':uzbasis, 'p':pbasis}
        coeff = {'up':upcoeff, 'um':umcoeff, 'uz':uzcoeff, 'p':pcoeff}
        griddata = {}
        for field, coeffs in coeff.items():
            data = np.zeros(Lmax*Nmax, dtype=vec.dtype)
            offset = 0
            for ell in range(Lmax):
                n = Nout(ell)
                data[ell*Nmax:ell*Nmax+n] = coeffs[offset:offset+n]
                offset += n
            basis = bases[field]
            griddata[field] = basis.expand(np.reshape(data, (Lmax,Nmax)))
        up, um, uz, p = griddata['up'], griddata['um'], griddata['uz'], griddata['p']
    else:
        up = upbasis.expand(np.reshape(upcoeff, (Lmax,Nmax)))
        um = umbasis.expand(np.reshape(umcoeff, (Lmax,Nmax)))
        uz = uzbasis.expand(np.reshape(uzcoeff, (Lmax,Nmax)))
        p  =  pbasis.expand(np.reshape( pcoeff, (Lmax,Nmax)))
    u, v, w = np.sqrt(0.5)*(up + um), -1j*np.sqrt(0.5)*(up - um), uz

    ns, neta = len(s), len(eta)
    ss, ee = s.reshape(1,ns), eta.reshape(neta,1)
    ur = ss * u + ee * np.sqrt(1-ss**2) * w 

    return u, v, w, p, tau, ur


def plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_fields):
    save_plots = False
    n = m+61
    num_modes = 6
    modes = list(zip([n]*num_modes, range(num_modes)))

    # Load the data
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    if save_plots:
        def save(fn, fig): save_figure(fn, fig)
    else:
        def save(fn, fig): pass

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
        ax.set_title(f'Inertial Wave Eigenvalues, $m$ = {m}')
        fig.set_tight_layout(True)

        filename = prefix + '-evalues-' + configstr + '.png'
        save(filename, fig)

    if not plot_fields:
        return

    # Get the target eigenpair
    ns, neta = 256, 257
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    fig, ax = plt.subplots(1,len(modes),figsize=(3*num_modes,5))
    for i, (n, ell) in enumerate(modes):
        # Compute the analytic eigenfrequency
        mode_index = (n,(n-m)//2-ell,m)
        evalue_target = 2*greenspan.compute_eigenvalues(mode_index[0], mode_index[2])[mode_index[1]-1]
        index = np.argmin(abs(evalues - evalue_target))
        evalue, evector = evalues[index], evectors[:,index]
    
        print('Plotting eigenvector with eigenvalue {:1.4f}'.format(evalue))
        u, v, w, p, tau, ur = expand_evectors(m, Lmax, Nmax, boundary_method, evector, s, eta)
   
        # Plot the pressure field
        Fgrid = p
        relative_real = np.linalg.norm(np.real(Fgrid))/np.linalg.norm(Fgrid)
        f = Fgrid.real if relative_real > 0.5 else Fgrid.imag
        sph.plotfield(s, eta, f, colorbar=False, fig=fig, ax=ax[i])
        ax[i].set_title(f'$\lambda = ${evalue_target:.4f}')
            
        # Compute the analytic pressure mode
        ss, ee = s[np.newaxis,:], eta[:,np.newaxis]
        zz = ee * np.sqrt(1 - ss**2)
        k = mode_index[1]
        panalytic = greenspan.compute_eigenmode(ss, zz, n, k, m)

        p /= np.max(abs(p))
        if np.max(abs(panalytic-p)) > np.max(abs(panalytic+p)):
            p *= -1
    
        # Compute the analytic eigenfrequency
        error_evalue = evalue_target - evalue
        print('    Eigenvalue error: ', error_evalue)

        error = panalytic - p.real
        print('    Eigenfunction error: ',np.max(abs(error)))

        # Spherical radial velocity component
        bc_error_top = np.max(np.abs(ur[-1,:]))
        bc_error_bot = np.max(np.abs(ur[ 0,:]))
        print('    Top    boundary error: {:1.3e}'.format(bc_error_top))
        print('    Bottom boundary error: {:1.3e}'.format(bc_error_bot))

        # Error plot
        plot_error = False
        if plot_error:
            sph.plotfield(s.ravel(), eta.ravel(), panalytic, colorbar=False)
            plt.title('analytic mode')
        
            sph.plotfield(s.ravel(), eta.ravel(), error)
            plt.title('error')

    filename = prefix + '-evectors-' + configstr + f'-n={n}-p.png'
    save(filename, fig)


def main():
    solve = True
    plot_evalues = True
    plot_fields = True

    m = 139
    Lmax, Nmax = 30, 42
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

