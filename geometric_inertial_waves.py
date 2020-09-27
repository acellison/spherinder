import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import spherinder as sph

from eigtools import eigsort
import os
import pickle
import greenspan_inertial_waves as greenspan


def remove_zero_rows(mat):
    rows, cols = mat.nonzero()
    zrows = list(set(range(np.shape(mat)[0])) - set(rows))
    if not zrows:
        return mat
    for z in zrows:
        i = np.argmax(rows > z)
        rows[i:] -= 1    
    return sparse.csr_matrix((mat.data, (rows,cols)), shape=(max(rows)+1,np.shape(mat)[1]))


def matrices(m, Lmax, Nmax, boundary_method):
    """Construct matrices for X = [i*u(+), i*u(-), i*w, p]
       FIXME: artificially truncating gradient and divergence
    """
    ncoeff = Lmax*Nmax
    Zero = sparse.lil_matrix((ncoeff,ncoeff))
    I = sparse.eye(ncoeff)

    # Scalar gradient operators
    Gradp, Gradm, Gradz = sph.Gradient()(m, Lmax, Nmax, alpha=0)
    Gradm = sph.resize(Gradm, Lmax, Nmax+1, Lmax, Nmax)
    Gradz = sph.resize(Gradz, Lmax-1, Nmax, Lmax, Nmax)

    # Vector divergence operator
    Div = sph.Divergence()(m, Lmax, Nmax, alpha=1)
    Div = sph.resize(Div, Lmax, Nmax+1, Lmax, Nmax)
    Divp = Div[:,:ncoeff]
    Divm = Div[:,ncoeff:2*ncoeff]
    Divz = Div[:,2*ncoeff:]

    # Boundary condition
    Rad = sph.RadialVector()(m, Lmax, Nmax, alpha=1)
    Boundary = sph.Boundary()(m, Lmax+1, Nmax+1, alpha=1, sigma=0)
    Bound = Boundary @ Rad
    Bound = remove_zero_rows(Bound)

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
    umats = [L00, L01, L02, L03]
    vmats = [L10, L11, L12, L13]
    wmats = [L20, L21, L22, L23]
    pmats = [L30, L31, L32, L33]

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([umats, vmats, wmats, pmats], format=sparse_format)

    # Boundary conditions
    def impenetrable(split_parity):
        ntau = np.shape(Bound)[0]
        row = sparse.hstack([Bound, sparse.lil_matrix((ntau,ncoeff))])

        # Tau contribution in the final ell coefficients
        whichtau = (2,3)
        connection = np.zeros((Lmax,1))
        connection[-1,-1] = 1.
        col1 = sparse.kron(connection, sparse.eye(Nmax))
        col1 = sparse.bmat([[(whichtau[j]==i)*col1 for j in range(len(whichtau))] for i in range(4)])

        # Tau contribution in final radial coefficient
        whichtau = (0,)
        connection = np.zeros((Nmax,1))
        connection[-1,-1] = 1.
        col2 = sparse.kron(sparse.eye(Lmax), connection)
        col2 = sparse.bmat([[(whichtau[j]==i)*col2 for j in range(len(whichtau))] for i in range(4)])

        col = sparse.hstack([col1,col2])

        return row, col

    # Create the boundary condition rows and tau columns
    row, col = impenetrable(split_parity=False)
    corner = np.zeros((np.shape(row)[0], np.shape(col)[1]))
    L = sparse.bmat([[  L, col],
                     [row, corner]], format='csr')

    M = sparse.bmat([[    M, 0*col],
                     [0*row, corner]], format='csr')

    M, L = M.tocsr(), L.tocsr()
    return M, L


def savedata(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def filename_prefix(directory='data'):
    basepath = os.path.join(os.path.dirname(__file__), directory)
    prefix = 'geometric_inertial_waves'
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
    print(tau)

    upbasis = [sph.psi(Nmax, m, ell, s, eta, sigma=+1, alpha=1) for ell in range(Lmax)]
    umbasis = [sph.psi(Nmax, m, ell, s, eta, sigma=-1, alpha=1) for ell in range(Lmax)]
    u0basis = [sph.psi(Nmax, m, ell, s, eta, sigma= 0, alpha=1) for ell in range(Lmax)]
    pbasis  = [sph.psi(Nmax, m, ell, s, eta, sigma= 0, alpha=0) for ell in range(Lmax)]

    # Get the grid space vector fields
    vec = vec.astype(np.complex128)
    vec[:3*ncoeff] /= 1j
    upcoeff = vec[:ncoeff] 
    umcoeff = vec[ncoeff:2*ncoeff] 
    wcoeff = vec[2*ncoeff:3*ncoeff]
    pcoeff = vec[3*ncoeff:4*ncoeff]

    # Convert to grid space
    up = sph.expand(upbasis, np.reshape(upcoeff, (Lmax,Nmax)))
    um = sph.expand(umbasis, np.reshape(umcoeff, (Lmax,Nmax)))
    w  = sph.expand(u0basis, np.reshape( wcoeff, (Lmax,Nmax)))
    p  = sph.expand( pbasis, np.reshape( pcoeff, (Lmax,Nmax)))
    u =       np.sqrt(0.5) * (up + um)
    v = -1j * np.sqrt(0.5) * (up - um)

    ns, neta = len(s), len(eta)
    ss, ee = s.reshape(1,ns), eta.reshape(neta,1)
    ur = np.sqrt(0.5) * ss * (up + um) + ee * np.sqrt(1-ss**2) * w

    return u, v, w, p, ur


def plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_slices, plot_fields):
    save_plots = True
    plot_field_indices = [3]
    mode_index = (20,9,1)

    # Load the data
    filename = pickle_filename(m, Lmax, Nmax, boundary_method)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    if save_plots:
        def savefig(fn): plt.savefig(fn)
    else:
        def savefig(_): pass

    evalue_target = 2*greenspan.compute_eigenvalues(mode_index[0], mode_index[2])[mode_index[1]-1]
    modestr = str(mode_index[0]) + str(mode_index[1]) + str(mode_index[2])
    configstr = 'm={}-Lmax={}-Nmax={}-{}'.format(m,Lmax,Nmax,boundary_method)
    prefix = filename_prefix('figures')

    nbad = len(np.where(np.abs(evalues.imag) > 1e-15)[0])
    print('Number of bad eigenvalues: {}/{}'.format(nbad,len(evalues)))

    # Plot the eigenvalues
    if plot_evalues:
        fig, ax = plt.subplots()
        ax.plot(evalues.real, evalues.imag, '.')
        ax.grid()
        ax.set_xlabel('Real(λ)')
        ax.set_ylabel('Imag(λ)')
        ax.set_title('Inertial Wave Eigenvalues in the Stretched Sphere')
        fig.set_tight_layout(True)

        filename = prefix + '-evalues-' + configstr + '.png'
        savefig(filename)

    if not (plot_slices or plot_fields):
        return

    # Get the target eigenpair
    index = np.argmin(abs(evalues - evalue_target))
    val, vec = evalues[index], evectors[:,index]

    print('Plotting eigenvector with eigenvalue {:1.4f}'.format(val))

    # Construct the basis polynomials
    ns, neta = 256, 255
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    u, v, w, p, ur = expand_evectors(m, Lmax, Nmax, boundary_method, vec, s, eta)

    # Spherical radial velocity component
    error_top = np.max(np.abs(ur[-1,:]))
    error_bot = np.max(np.abs(ur[ 0,:]))
    if error_top > 1.5e-12 or error_bot > 1.5e-12:
        print('Top    boundary error: {:1.3e}'.format(error_top))
        print('Bottom boundary error: {:1.3e}'.format(error_bot))
#    assert np.max(abs(ur[-1,:])) < 1.5e-12
#    assert np.max(abs(ur[ 0,:])) < 1.5e-12

    fields = [u,v,w,p,ur]
    field_names = ['u','v','w','p','u_r']

    if plot_slices:
        # Spherical radial velocity at the boundary
        fig, radial_axes = plt.subplots()
        radial_axes.plot(s, np.abs(ur[-1,:]), label='top')
        radial_axes.plot(s, np.abs(ur[ 0,:]), label='bottom')
        radial_axes.set_title('spherical radial velocity component')
        radial_axes.set_xlabel('s')
        radial_axes.legend()
        radial_axes.grid()

    if not plot_fields:
        plot_field_indices = []

    for i in range(len(plot_field_indices)):
        field_index = plot_field_indices[i]
        Fgrid = fields[field_index]
        relative_real = np.linalg.norm(np.real(Fgrid))/np.linalg.norm(Fgrid)
        f = Fgrid.real if relative_real > 0.5 else Fgrid.imag

        sph.plotfield(s, eta, f)
        plt.title(r'${}$'.format(field_names[field_index]))
        filename = prefix + '-evector-' + configstr + '-' + modestr + '-' + field_names[field_index] + '.png'
        savefig(filename)


def main():
    solve = True
    plot_evalues = True
    plot_fields = True
    plot_slices = False

    m = 1
    Lmax, Nmax = 16, 16
    boundary_method = 'tau'

    print('Inertial Waves, m = {}'.format(m))
    print('  Domain size: Lmax = {}, Nmax = {}'.format(Lmax, Nmax))

    if solve:
        solve_eigenproblem(m, Lmax, Nmax, boundary_method)

    if plot_fields or plot_evalues or plot_slices:
        plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_slices, plot_fields)
        plt.show()


if __name__=='__main__':
    main()

