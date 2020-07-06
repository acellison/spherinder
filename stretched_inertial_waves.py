import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from dedalus_sphere import jacobi as Jacobi
from dedalus_sphere import zernike as Zernike
from kronecker import Kronecker
from eigtools import eigsort
import os
import pickle
import greenspan_inertial_waves as greenspan


# Jacobi operators
A = Jacobi.operator('A')
B = Jacobi.operator('B')
DZ = Jacobi.operator('D')
Z = Jacobi.operator('Z')
IdZ = Jacobi.operator('Id')

# Zernike operators
dim = 2
DR = Zernike.operator(dim, 'D')
E = Zernike.operator(dim, 'E')
R = Zernike.operator(dim, 'R')
IdR = Zernike.operator(dim, 'Id')

# Composite Jacobi operators
ZdZ = Z @ DZ(+1)                # Z * d/dZ
AB = A(+1) @ B(+1)              # a,b -> a+1,b+1

# Composite Zernike operators
H = IdR - R(-1) @ R(+1)         # 1 - R**2


g_galerkin_location = 'first'
#g_galerkin_location = 'last'


def make_square_resizer(dimensions):
    def resizer(dimension, mat):
        n = dimensions[dimension]
        mat = mat[:n, :n]
        ncols = np.shape(mat)[1]
        if ncols < n:
            mat = sparse.hstack([mat, np.zeros((n, n - ncols))])
        return mat
    return resizer


def galerkin_matrix_2d(Nmax, k, m, Lmax, a, b, which):
    Sr = sparse.eye(Nmax)
    bc = Jacobi.polynomials(Lmax,a,b,1.0)
    if which == 'first':
        # D_0 = P_0
        # D_1 = P_1
        # D_n = P_{n-2}(1) * P_n - P_n(1) * P_{n-2},  n >= 2
        Sz = sparse.diags([np.concatenate(([1,1],bc[:-2])), -bc[2:]], [0, 2])
    elif which == 'last':
        # D_n = P_n(1) * P_{n+2} - P_{n-2}(1) * P_n,  n < L-2
        # D_{L-2} = P_{L-2}
        # D_{L-1} = P_{L-1}
        Sz = sparse.diags([np.concatenate((bc[2:],[1,1])), -bc[:-2]], [0, -2])
    else:
        raise ValueError('Unsupported basis recombination mode')
    return sparse.kron(Sr, Sz)


def galerkin_matrix(Nmax, k, m, Lmax, a, b, which=g_galerkin_location):
    Sp = galerkin_matrix_2d(Nmax, k, m+1, Lmax, a, b, which=which)
    Sm = galerkin_matrix_2d(Nmax, k, m-1, Lmax, a, b, which=which)
    S0 = galerkin_matrix_2d(Nmax, k, m,   Lmax, a, b, which=which)
    I = sparse.eye(Nmax*Lmax)
    return sparse.block_diag([Sp,Sm,S0,I])



def matrices(Nmax, k, m, Lmax, a, b, boundary_method):
    """Construct matrices for X = [i*u(+), i*u(-), i*h*w, p]
    """
    resizer = make_square_resizer([Nmax, Lmax])
    kron = lambda ops, opz: Kronecker([ops, opz], resizer=resizer)

    Zero = 0.*kron(IdR, IdZ)
    xi = np.sqrt(0.5)

    # Time derivative matrices
    M00 = kron(H, AB)
    M11 = kron(H, AB)
    M22 = kron(IdR, AB)
    M33 = Zero

    # i*u+ equation - spin+ velocity component
    L00 = -2 * kron(H, AB)
    L01 = Zero
    L02 = Zero
    L03 = -xi * (kron(E(-1) @ DR(+1), AB) + kron(R(+1), ZdZ))

    # i*u- equation - spin- velocity component
    L10 = Zero
    L11 =  2 * kron(H, AB)
    L12 = Zero
    L13 = -xi * (kron(E(-1) @ DR(-1), AB) + kron(R(-1), ZdZ))

    # i*h*w equation - vertical velocity component
    L20 = Zero
    L21 = Zero
    L22 = Zero
    L23 = -kron(IdR, DZ(+1))

    # Divergence equation
    L30 = xi * (kron(E(-1) @ DR(-1), AB) + kron(R(-1), ZdZ))
    L31 = xi * (kron(E(-1) @ DR(+1), AB) + kron(R(+1), ZdZ))
    L32 = kron(IdR, DZ(+1))
    L33 = Zero

    verbose = True
    if verbose:
        print_codomains = lambda name, mats: [print('{}{}: {}'.format(name,i,mats[i].codomains)) for i in range(4)]
        print_codomains('M:      ', [M00, M11, M22, M33]); print()
        print_codomains('i*u(+): ', [L00, L01, L02, L03]); print()
        print_codomains('i*u(-): ', [L10, L11, L12, L13]); print()
        print_codomains('i*h*w:  ', [L20, L21, L22, L23]); print()
        print_codomains('p:      ', [L30, L31, L32, L33]);

    # Construct the matrices
    uparams = [(Nmax,k,m+1),(Lmax,a,b)]  # i * u(+)
    vparams = [(Nmax,k,m-1),(Lmax,a,b)]  # i * u(-)
    wparams = [(Nmax,k,m),  (Lmax,a,b)]  # i * h * w
    pparams = [(Nmax,k,m),  (Lmax,a,b)]  # p
    param_sets = [uparams, vparams, wparams, pparams]
    make_matrices = lambda mats: [mats[i](param_sets[i]) for i in range(4)]

    Mmats = make_matrices([M00, M11, M22, M33])
    umats = make_matrices([L00, L01, L02, L03])
    vmats = make_matrices([L10, L11, L12, L13])
    wmats = make_matrices([L20, L21, L22, L23])
    pmats = make_matrices([L30, L31, L32, L33])

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([umats, vmats, wmats, pmats], format=sparse_format)

    # Boundary conditions
    ncoeff = Nmax*Lmax
    def truncate_radial(index, count=1):
        nc = count*Lmax
        row = (index+1)*ncoeff - nc
        M[row:row+nc,:] = 0
        L[row:row+nc,:] = 0
        L[row:row+nc,row:row+nc] = sparse.eye(nc)

    def impenetrable(split_parity):
        etas = np.array([1.,-1.])
        bc = Jacobi.polynomials(Lmax,a,b,etas,normalised=True).T
        Su = xi * R(-1)(Nmax,k,m+1)[:Nmax,:].todense()
        Sv = xi * R(+1)(Nmax,k,m-1).todense()

        if split_parity:
            bcmat = 0*bc
            bcmat[0,:] = 1/2*(bc[0,:] + bc[1,:])
            bcmat[1,:] = 1/2*(bc[0,:] - bc[1,:])
            bcmat[np.abs(bcmat) < 1e-12] = 0
            bcmatw = np.flipud(bcmat)
        else:
            bcmat = bc
            bcmatw = np.reshape(etas,(2,1))*bcmat

        bcu = np.kron(Su, bcmat)
        bcv = np.kron(Sv, bcmat)
        bcw = np.kron(np.eye(Nmax), bcmatw)
        row = np.concatenate([bcu, bcv, bcw, np.zeros((2*Nmax,ncoeff))], axis=1)
       
        whichtau = (2,3)
        if whichtau[0] == whichtau[1]:
            # Tau errors in the final two coefficients of the divergence equation
            connection = AB(Lmax,a,b)[:,-2:]
            col = sparse.kron(IdR(Nmax,k,m), connection)
            col = sparse.bmat([[(whichtau[0]==i)*col] for i in range(4)])
        else:
            # Tau contribution in the final coefficient of w and divergence equations
            connection = AB(Lmax,a,b)[:,-1:]
            col = sparse.kron(IdR(Nmax,k,m), connection)
            col = sparse.bmat([[(whichtau[0]==i)*col, (whichtau[1]==i)*col] for i in range(4)])

        return row, col

    # Final radial coefficient of u(+) interferes with boundary
    truncate_radial(0)

    # Create the boundary condition rows and tau columns
    row, col = impenetrable(split_parity=True)
    corner = np.zeros((np.shape(row)[0], np.shape(col)[1]))
    L = sparse.bmat([[  L, col],
                     [row, corner]], format='csr')

    M = sparse.bmat([[    M, 0*col],
                     [0*row, corner]], format='csr')

    if boundary_method == 'galerkin':
        S = galerkin_matrix(Nmax, k, m, Lmax, a, b)
        L[:,:4*ncoeff] = L[:,:4*ncoeff] @ S
        M[:,:4*ncoeff] = M[:,:4*ncoeff] @ S

        L[np.abs(L) < 1e-12] = 0
        M[np.abs(M) < 1e-12] = 0
        L.eliminate_zeros()
        M.eliminate_zeros()

    M, L = M.tocsr(), L.tocsr()
    return M, L


def tobanded(A, Nmax, Lmax):
    ncoeff = Nmax*Lmax
    neq = np.shape(A)[0]

    baseinds = np.array([0,1,2,3])*ncoeff
    inds = [] 
    tauind = 4*ncoeff
    for i in range(ncoeff):
        if i % Lmax == 0:
            inds += [tauind, tauind+1]
            tauind += 2
        inds += (baseinds + i).tolist()

    B = A[inds,:]
    C = B[:,inds]
    return C


def savedata(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def filename_prefix(directory='data'):
    basepath = os.path.join(os.path.dirname(__file__), directory)
    prefix = 'stretched_inertial_waves'
    return os.path.join(basepath, os.path.join(prefix, prefix))


def pickle_filename(m, k, Nmax, Lmax, boundary_method, directory='data'):
    return filename_prefix(directory) + '-evalues-m={}-k={}-Nmax={}-Lmax={}-{}.pckl'.format(m,k,Nmax,Lmax,boundary_method)


def solve_eigenproblem(m, k, Nmax, Lmax, boundary_method):
    a, b = 0, 0  # Legendre polynomials since η doesn't appear in volume element dV = (1-s**2)**1/2 s ds dφ dη

    # Construct the system
    M, L = matrices(Nmax, k, m, Lmax, a, b, boundary_method)

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
    data = {'Nmax': Nmax, 'k': k, 'm': m, 
            'Lmax': Lmax, 'a': a, 'b': b,
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors}
    filename = pickle_filename(m, k, Nmax, Lmax, boundary_method)
    savedata(filename, data)


def plotfield(s, eta, f, zoombox=None):
    ss = np.reshape(s,(1,len(s)))
    ee = np.reshape(eta,(len(eta),1))
    zz = np.sqrt(1-ss**2)*ee

    fig, ax = plt.subplots(figsize=(4.25,6))
    im = ax.pcolormesh(ss, zz, f, cmap='RdBu')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('s')
    ax.set_ylabel('z')
    if zoombox is None:
        ax.set_aspect('equal', adjustable='box')
    else:
        ax.set_xlim(zoombox[0])
        ax.set_ylim(zoombox[1])
    fig.set_tight_layout(True)


def expansion(F, T, Q):
    return T @ (Q @ F).T


def expand_evectors(Nmax, k, m, Lmax, a, b, boundary_method, vec, s, eta):
    z = 2*s**2 - 1
    Tn = Jacobi.polynomials(Lmax,a,b,eta).T
    Qn = Zernike.polynomials(dim,Nmax,k,m,z).T
    Qnp = Zernike.polynomials(dim,Nmax,k,m+1,z).T
    Qnm = Zernike.polynomials(dim,Nmax,k,m-1,z).T

    ncoeff = Nmax*Lmax
    tau = vec[4*ncoeff:]
    print(tau)

    if boundary_method == 'galerkin':
        S = galerkin_matrix(Nmax, k, m, Lmax, a, b)
        vec = S @ vec[:4*ncoeff]

    # Get the grid space vector fields
    vec = vec.astype(np.complex128)
    vec[:3*ncoeff] /= 1j
    upcoeff = vec[:ncoeff] 
    umcoeff = vec[ncoeff:2*ncoeff] 
    Wcoeff = vec[2*ncoeff:3*ncoeff]
    pcoeff = vec[3*ncoeff:4*ncoeff]

    # Convert to grid space
    up = expansion(np.reshape(upcoeff, (Nmax,Lmax)), Tn, Qnp)
    um = expansion(np.reshape(umcoeff, (Nmax,Lmax)), Tn, Qnm)
    u =       np.sqrt(0.5) * (up + um)
    v = -1j * np.sqrt(0.5) * (up - um)
    W = expansion(np.reshape(Wcoeff, (Nmax,Lmax)), Tn, Qn)
    p = expansion(np.reshape(pcoeff, (Nmax,Lmax)), Tn, Qn)

    ns, neta = len(s), len(eta)
    scale = np.reshape(np.sqrt(1-s**2), (1,ns))
    scale[scale == 0.] = 1.
    w = W / scale

    ur = np.reshape(s,(1,ns))*u + np.reshape(eta,(neta,1))*W

    return u, v, w, p, ur


def plot_solution(m, k, Nmax, Lmax, boundary_method, plot_evalues, plot_slices, plot_fields):
    save_plots = True
    plot_field_indices = [3]
    mode_index = (40,1,30)    

    # Load the data
    filename = pickle_filename(m, k, Nmax, Lmax, boundary_method)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    k, a, b = data['k'], data['a'], data['b']
    evalues, evectors = data['evalues'], data['evectors']

    if save_plots:
        def savefig(fn): plt.savefig(fn)
    else:
        def savefig(_): pass

    evalue_target = greenspan.compute_eigenvalues(mode_index[0], mode_index[2])[mode_index[1]-1]
    modestr = str(mode_index[0]) + str(mode_index[1]) + str(mode_index[2])
    configstr = 'm={}-k={}-Nmax={}-Lmax={}-{}'.format(m,k,Nmax,Lmax,boundary_method)
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
    ns, neta = 1000, 1001
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    u, v, w, p, ur = expand_evectors(Nmax, k, m, Lmax, a, b, boundary_method, vec, s, eta)

    # Spherical radial velocity component
    error_top = np.max(np.abs(ur[-1,:]))
    error_bot = np.max(np.abs(ur[ 0,:]))
    if error_top > 1.5e-12 or error_bot > 1.5e-12:
        print('Top    boundary error: {:1.3e}'.format(error_top))
        print('Bottom boundary error: {:1.3e}'.format(error_bot))
    assert np.max(abs(ur[-1,:])) < 1.5e-12
    assert np.max(abs(ur[ 0,:])) < 1.5e-12

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

        plotfield(s, eta, f)
        plt.title(r'${}$'.format(field_names[field_index]))
        filename = prefix + '-evector-' + configstr + '-' + modestr + '-' + field_names[field_index] + '.png'
        savefig(filename)


def main():
    solve = False
    plot_evalues = False
    plot_fields = True
    plot_slices = False

    m, k = 30, 0
    Nmax, Lmax = 24, 24
    boundary_method = 'tau'

    print('Inertial Waves, m = {}'.format(m))
    print('  Domain size: Nmax = {}, Lmax = {}'.format(Nmax, Lmax))

    if solve:
        solve_eigenproblem(m, k, Nmax, Lmax, boundary_method)

    if plot_fields or plot_evalues or plot_slices:
        plot_solution(m, k, Nmax, Lmax, boundary_method, plot_evalues, plot_slices, plot_fields)
        plt.show()


if __name__=='__main__':
    main()

