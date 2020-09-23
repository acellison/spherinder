import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from dedalus_sphere import jacobi as Jacobi
from eigtools import eigsort
import os
import pickle
import greenspan_inertial_waves as greenspan


def psi(Nmax, m, ell, s, eta, sigma=0, alpha=0, beta=0):
    """Basis function for our fields"""
    ns, neta = len(s), len(eta)
    t = 2*s**2 - 1

    Peta = Jacobi.polynomials(ell+1,alpha,alpha,eta)[-1,:].reshape(neta,1)
    Ps = Jacobi.polynomials(Nmax,ell+alpha+beta+1/2,m+sigma,t)
    tt = t.reshape(1,ns)
    return [Peta * (1+tt)**((m+sigma)/2) * (1-tt)**(ell/2) * Ps[k,:] for k in range(Nmax)]


def expand(basis, coeffs):
    """Expand the coefficient vector to grid space"""
    f = np.zeros(np.shape(basis[0][0]), dtype=np.complex128)
    for ell in range(len(basis)):
        for k in range(len(basis[ell])):
            f += coeffs[ell,k] * basis[ell][k]
    return f


def plotfield(s, eta, f, fig=None, ax=None, stretch=False):
    s, eta = s.ravel(), eta.ravel()
    ss = np.reshape(s,(1,len(s)))
    ee = np.reshape(eta,(len(eta),1))
    zz = np.sqrt(1-ss**2)*ee
    y = ee if stretch else zz
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4.25,6))
    im = ax.pcolormesh(ss, y, f, cmap='RdBu')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('s')
    if stretch:
        ax.set_ylabel('η')
    else:
        ax.set_ylabel('z')
    ax.set_aspect('equal', adjustable='box')
    fig.set_tight_layout(True)


# Jacobi operators
A = Jacobi.operator('A')
B = Jacobi.operator('B')
C = Jacobi.operator('C')
D = Jacobi.operator('D')
Z = Jacobi.operator('Z')
Id = Jacobi.operator('Id')

# Composite Jacobi operators
AB = A(+1) @ B(+1)   # a,b -> a+1,b+1


def make_operator(zmat, smats, Lmax=None, Nmax=None):
    """Kronecker out an operator.  Since the radial operators depend on ell,
       we require a separate operator matrix for each vertical expansion coefficient.
       FIXME: need to maintain square shapes for our operators"""
    def pad(mat,n):
        if np.isscalar(mat):       return mat
        if np.shape(mat) == (1,1): return mat
        mat = mat[:n,:]
        nrows, ncols = np.shape(mat)
        if nrows < n:
            print('padding')
            mat = sparse.vstack([mat, np.zeros((n-nrows, n))])
        return mat

    Lout, Lin = np.shape(zmat)
    Nout, Nin = 0, 0
    for smat in smats:
        if np.ndim(smat) < 2: continue
        sh = np.shape(smat)
        if Nout < sh[0]: Nout = sh[0]
        if Nin  < sh[1]: Nin  = sh[1]

    if Lmax is not None:
        Lout = Lmax
        zmat = pad(zmat, Lmax)
    if Nmax is not None:
        Nout = Nmax
        smats = [pad(smat, Nmax) for smat in smats]

    # Construct the operator matrix
    op = sparse.lil_matrix((Lout*Nout,Lin*Nin))
    rows, cols = zmat.nonzero()
    for row, col in zip(rows, cols):
        value = zmat[row,col]
        if not np.isscalar(value):
            value = value.todense()[0,0]
        if np.abs(value) < 1e-15:
            continue
        smat = smats[col]
        op[Nout*row:Nout*(row+1),Nin*col:Nin*(col+1)] = value * smat
    return op


def make_conversion_operator(m, Lmax, Nmax, alpha, sigma, Ntrunc=None):
    """Convert up in alpha index"""
    if Ntrunc is None:
        Ntrunc = Nmax

    opz = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()                   # (ell,alpha,alpha) -> (ell,alpha+1,alpha+1)
    alpha_ell = np.diag(opz)
    beta_ell = np.diag(opz,2)

    zmat = np.diag(alpha_ell)
    smats = [A(+1)(Nmax,ell+alpha+1/2,m+sigma) for ell in range(Lmax)]  # (n,a,b) -> (n,a+1,b)
    Op1 = make_operator(zmat, smats, Lmax, Ntrunc)

    zmat = np.diag(beta_ell,2)
    smats = [A(-1)(Nmax,ell+alpha+1/2,m+sigma) for ell in range(Lmax)]  # (n,a,b) -> (n+1,a-1,b)
    Op2 = make_operator(zmat, smats, Lmax, Ntrunc)

    Op = Op1 + Op2
    return Op


def make_radial_operator(m, Lmax, Nmax, alpha, Ntrunc=None):
    """Extract the spherical radial part of a velocity field"""   
    if Ntrunc is None:
        Ntrunc = Nmax

    # Coeff space operator: s * u(+)
    zmat = Id(Lmax,alpha,alpha)
    smats = [B(-1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]                 # (n,a,b) -> (n+1,a,b-1)
    Opp = 1/2 * make_operator(zmat, smats, Lmax, Ntrunc)

    # Coeff space operator: s * u(-)
    zmat = Id(Lmax,alpha,alpha)
    smats = [B(+1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]                 # (n,a,b) -> (n,a,b+1)
    Opm = 1/2 * make_operator(zmat, smats, Lmax, Ntrunc)

    # Coeff space operator: z * w = eta * (1-s**2)*0.5 * w
    opz = Z(Lmax,alpha,alpha).todense()

    zmat = np.diag(np.diag(opz,-1),-1)[:,:Lmax]
    smats = [A(+1)(Nmax+1,ell+alpha+1/2,m)[:,:Nmax] for ell in range(Lmax)] + [0]  # (n,a,b) -> (n,a+1,b)
    Opz1 = make_operator(zmat, smats, Lmax, Ntrunc)

    zmat = np.diag(np.diag(opz,+1),+1)
    smats = [0] + [A(-1)(Nmax,ell+alpha+1/2,m) for ell in range(1,Lmax)]           # (n,a,b) -> (n+1,a-1,b)
    Opz2 = make_operator(zmat, smats, Lmax, Ntrunc)

    Opz = 1/np.sqrt(2) * (Opz1 + Opz2)

    return sparse.hstack([Opp, Opm, Opz])


def make_r_multiplication_operator(m, Lmax, Nmax, alpha, Ntrunc=None):
    """Compute r e_r multiplication"""
    if Ntrunc is None:
        Ntrunc = Nmax+1

    # u(+) operator
    zmat = Id(Lmax,alpha,alpha)
    smats = [B(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]               # (n,a,b) -> (n,a,b+1)
    Opp = 1/2 * make_operator(zmat, smats, Lmax, Nmax)

    # u(-) operator
    zmat = Id(Lmax,alpha,alpha)
    smats = [B(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]               # (n,a,b) -> (n+1,a,b-1)
    Opm = 1/2 * make_operator(zmat, smats, Lmax, Nmax+1)

    # u(z) operator
    opz = Z(Lmax,alpha,alpha).todense()                                        # (ell,alpha,alpha) -> (ell+1,alpha,alpha)

    zmat = np.diag(np.diag(opz,-1),-1)[:,:Lmax]
    smats = [A(+1)(Nmax+1,ell+alpha+1/2,m)[:,:Nmax] for ell in range(Lmax+1)]  # (n,a,b) -> (n,a+1,b)
    Opz1 = make_operator(zmat, smats, Lmax=None, Nmax=Nmax+1)

    zmat = np.diag(np.diag(opz,+1),+1)
    smats = [0] + [A(-1)(Nmax,ell+alpha+1/2,m) for ell in range(1,Lmax+1)]     # (n,a,b) -> (n+1,a-1,b)
    Opz2 = make_operator(zmat, smats, Lmax=None, Nmax=Nmax+1)
    Opz2 = sparse.vstack([Opz2,sparse.lil_matrix((Nmax+1,Lmax*Nmax))])

    Opz = 1/np.sqrt(2) * (Opz1 + Opz2)

    Cp = make_conversion_operator(m, Lmax, Nmax=Nmax, Ntrunc=Nmax+1, alpha=alpha, sigma=+1)
    Cm = make_conversion_operator(m, Lmax, Nmax=Nmax+1, Ntrunc=Nmax+2, alpha=alpha, sigma=-1)
    Cz = make_conversion_operator(m, Lmax+1, Nmax=Nmax+1, Ntrunc=Nmax+2, alpha=alpha, sigma= 0)

    return [Cp @ Opp, Cm @ Opm, Cz @ Opz], [(Lmax,Nmax+1),(Lmax,Nmax+2),(Lmax+1,Nmax+2)]


def make_gradient_operator(m, Lmax, Nmax, alpha, Ntrunc=None):
    """Compute the gradient of a scalar field"""
    if Ntrunc is None:
        Ntrunc = Nmax

    op = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()
    alpha_ell = np.diag(op)
    beta_ell = -np.diag(op,2)
    
    # e(+)^* . Grad
    zmat = np.diag(2*alpha_ell)
    smats = [D(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
    Op1 = make_operator(zmat, smats, Lmax, Ntrunc)
    zmat = np.diag(2*beta_ell,2)
    smats = [C(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
    Op2 = make_operator(zmat, smats, Lmax, Ntrunc)
    Opp = Op1 + Op2
    
    # e(-)^* . Grad
    zmat = np.diag(2*alpha_ell)
    smats = [C(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
    Op1 = make_operator(zmat, smats, Lmax, Ntrunc)
    zmat = np.diag(2*beta_ell,2)
    smats = [D(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
    Op2 = make_operator(zmat, smats, Lmax, Ntrunc)
    Opm = Op1 + Op2

    # e(z)^* . Grad
    zmat = np.sqrt(2) * D(+1)(Lmax,alpha,alpha)
    smats = [Id(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
    Opz = make_operator(zmat, smats, Lmax, Ntrunc)

    return sparse.vstack([Opp, Opm, Opz])


def make_divergence_operator(m, Lmax, Nmax, alpha, Ntrunc=None):
    """Compute the divergence of a vector field"""
    if Ntrunc is None:
        Ntrunc = Nmax

    op = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()
    gamma_ell = np.diag(op)
    delta_ell = -np.diag(op,2)
   
    # Div . e(+)^* .
    zmat = np.diag(2*gamma_ell)
    smats = [C(+1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]
    Op1 = make_operator(zmat, smats, Lmax, Ntrunc)
    zmat = np.diag(2*delta_ell,2)
    smats = [D(-1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]
    Op2 = make_operator(zmat, smats, Lmax, Ntrunc)
    Opp = Op1 + Op2

    # Div . e(-)^* . 
    zmat = np.diag(2*gamma_ell)
    smats = [D(+1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]
    Op1 = make_operator(zmat, smats, Lmax, Ntrunc)
    zmat = np.diag(2*delta_ell,2)
    smats = [C(-1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]
    Op2 = make_operator(zmat, smats, Lmax, Ntrunc)
    Opm = Op1 + Op2
 
    # Div . e(z)^* .
    zmat = np.sqrt(2) * D(+1)(Lmax,alpha,alpha)
    smats = [Id(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
    Opz = make_operator(zmat, smats, Lmax, Ntrunc)

    return sparse.hstack([Opp, Opm, Opz])


def make_curl_operator(m, Lmax, Nmax, alpha, Ntrunc=None):
    """Compute the curl of a vector field"""
    if Ntrunc is None:
        Ntrunc = Nmax

    op = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()
    gamma_ell = np.diag(op)
    delta_ell = -np.diag(op,2)
   
    # e(+)^* . Curl
    zmat = 1j * np.sqrt(2) * D(+1)(Lmax,alpha,alpha)
    smats = [Id(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]
    Opp_p = make_operator(zmat, smats, Lmax, Ntrunc)

    zmat = -2j * np.diag(gamma_ell)
    smats = [D(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
    Op1 = make_operator(zmat, smats, Lmax, Ntrunc)
    zmat = -2j * np.diag(delta_ell,2)
    smats = [C(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
    Op2 = make_operator(zmat, smats, Lmax, Ntrunc)
    Opp_z = Op1 + Op2

    # e(-)^* . Curl
    zmat = -1j * np.sqrt(2) * D(+1)(Lmax,alpha,alpha)
    smats = [Id(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]
    Opm_m = make_operator(zmat, smats, Lmax, Ntrunc)

    zmat = 2j * np.diag(gamma_ell)
    smats = [C(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
    Op1 = make_operator(zmat, smats, Lmax, Ntrunc)
    zmat = 2j * np.diag(delta_ell,2)
    smats = [D(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
    Op2 = make_operator(zmat, smats, Lmax, Ntrunc)
    Opm_z = Op1 + Op2
 
    # e(z)^* . Curl
    zmat = -2j * np.diag(gamma_ell)
    smats = [C(+1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]
    Op1 = make_operator(zmat, smats, Lmax, Ntrunc)
    zmat = -2j * np.diag(delta_ell,2)
    smats = [D(-1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]
    Op2 = make_operator(zmat, smats, Lmax, Ntrunc)
    Opz_p = Op1 + Op2

    zmat = 2j * np.diag(gamma_ell)
    smats = [D(+1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]
    Op1 = make_operator(zmat, smats, Lmax, Ntrunc)
    zmat = 2j * np.diag(delta_ell,2)
    smats = [C(-1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]
    Op2 = make_operator(zmat, smats, Lmax, Ntrunc)
    Opz_m = Op1 + Op2

    return sparse.bmat([[  Opp_p, 0*Opp_p,   Opp_z],
                        [0*Opm_m,   Opm_m,   Opm_z],
                        [  Opz_p,   Opz_m, 0*Opz_p]])


def make_scalar_laplacian_operator(m, Lmax, Nmax, alpha):
    Grad = make_gradient_operator(m, Lmax, Nmax, alpha, Ntrunc=Nmax+1)
    Div = make_divergence_operator(m, Lmax, Nmax+1, alpha+1, Ntrunc=Nmax)
    return Div @ Grad


def make_vector_laplacian_operator(m, Lmax, Nmax, alpha, incompressible=False):
    Curl1 = make_curl_operator(m, Lmax, Nmax, alpha, Ntrunc=Nmax+1)
    Curl2 = make_curl_operator(m, Lmax, Nmax+1, alpha+1, Ntrunc=Nmax)
    Lap = -Curl2 @ Curl1
    Lap = Lap.real

    if not incompressible:
        Div = make_divergence_operator(m, Lmax, Nmax, alpha, Ntrunc=Nmax+1)
        Grad = make_gradient_operator(m, Lmax, Nmax+1, alpha+1, Ntrunc=Nmax)
        Lap += Grad @ Div

    return Lap


def _make_boundary_evaluation_operator_impl(m, Lmax, Nmax, alpha, sigma):
    """Compute the boundary evaluation operator, split into the even and odd ell indices"""
    L = Lmax-1
    even_conversions = [(A(+1)**(L//2-ell) @ A(-1)**ell)(Nmax,2*ell+alpha+1/2,m+sigma) for ell in range(L//2+1)]
    odd_conversions = [(A(+1)**(((L-1)//2)-ell) @ A(-1)**ell)(Nmax,2*ell+1+alpha+1/2,m+sigma) for ell in range((L+1)//2)]

    bc = Jacobi.polynomials(Lmax,alpha,alpha,1.)

    Opeven = sparse.lil_matrix((Nmax+L//2,Lmax*Nmax))
    Opodd  = sparse.lil_matrix((Nmax+(L-1)//2,Lmax*Nmax))
    for ell in range(Lmax):
        if ell % 2 == 0:
            op, mat = even_conversions, Opeven
        else:
            op, mat = odd_conversions, Opodd        
        op = bc[ell] * op[ell//2]
        mat[:np.shape(op)[0],ell*Nmax:(ell+1)*Nmax] = op

    Op = sparse.vstack([Opeven,Opodd])
    return Op


def make_boundary_evaluation_operator(m, Lmax, Nmax, alpha, sigma):
    make_op = lambda s: _make_boundary_evaluation_operator_impl(m, Lmax, Nmax, alpha, s)
    if np.isscalar(sigma):
        return make_op(sigma)
    else:
        return sparse.hstack([make_op(s) for s in sigma])


def matrices(m, Lmax, Nmax, boundary_method):
    """Construct matrices for X = [i*u(+), i*u(-), i*w, p]
    """
    ncoeff = Lmax*Nmax
    Zero = sparse.lil_matrix((ncoeff,ncoeff))
    I = sparse.eye(ncoeff)

    Grad = make_gradient_operator(m, Lmax, Nmax, alpha=0)
    Div = make_divergence_operator(m, Lmax, Nmax, alpha=1)
    Laps = make_scalar_laplacian_operator(m, Lmax, Nmax, alpha=0)
    Lapv = make_vector_laplacian_operator(m, Lmax, Nmax, alpha=1, incompressible=True)

    truncate_boundary = False
    Boundary = make_boundary_evaluation_operator(m, Lmax, Nmax+int(not truncate_boundary), alpha=1, sigma=[+1,-1,0])

    # u(+) conversion from alpha=1 to alpha=3
    C1p = make_conversion_operator(m, Lmax, Nmax, Ntrunc=Nmax+1, alpha=1, sigma=+1)
    C2p = make_conversion_operator(m, Lmax, Nmax, Ntrunc=Nmax,   alpha=2, sigma=+1)
    Cp = C2p @ C1p

    # u(-) conversion from alpha=1 to alpha=3
    C1m = make_conversion_operator(m, Lmax, Nmax, Ntrunc=Nmax+1, alpha=1, sigma=-1)
    C2m = make_conversion_operator(m, Lmax, Nmax, Ntrunc=Nmax,   alpha=2, sigma=-1)
    Cm = C2p @ C1p

    # u(z) conversion from alpha=1 to alpha=3
    C1z = make_conversion_operator(m, Lmax, Nmax, Ntrunc=Nmax+1, alpha=1, sigma=0)
    C2z = make_conversion_operator(m, Lmax, Nmax, Ntrunc=Nmax,   alpha=2, sigma=0)
    Cz = C2z @ C1z

    # scalar conversion from alpha=0 to alpha=2
    C1s = make_conversion_operator(m, Lmax, Nmax, Ntrunc=Nmax+1, alpha=0, sigma=0)
    C2s = make_conversion_operator(m, Lmax, Nmax, Ntrunc=Nmax,   alpha=1, sigma=0)
    Cs = C2T @ C1z

    # Time derivative matrices
    M00 = Cp
    M11 = Cm
    M22 = Cz
    M33 = Zero
    M44 = Cs

    # i*u+ equation - spin+ velocity component
    L00 = -2 * Cp
    L01 = Zero
    L02 = Zero
    L03 = -Gradp

    # i*u- equation - spin- velocity component
    L10 = Zero
    L11 = 2 * Cm
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
        ntau = np.shape(Boundp)[0]
        row = sparse.hstack([Boundp, Boundm, Boundz, sparse.lil_matrix((ntau,ncoeff))])

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

    upbasis = [psi(Nmax, m, ell, s, eta, sigma=+1, alpha=1) for ell in range(Lmax)]
    umbasis = [psi(Nmax, m, ell, s, eta, sigma=-1, alpha=1) for ell in range(Lmax)]
    u0basis = [psi(Nmax, m, ell, s, eta, sigma= 0, alpha=1) for ell in range(Lmax)]
    pbasis  = [psi(Nmax, m, ell, s, eta, sigma= 0, alpha=0) for ell in range(Lmax)]

    # Get the grid space vector fields
    vec = vec.astype(np.complex128)
    vec[:3*ncoeff] /= 1j
    upcoeff = vec[:ncoeff] 
    umcoeff = vec[ncoeff:2*ncoeff] 
    wcoeff = vec[2*ncoeff:3*ncoeff]
    pcoeff = vec[3*ncoeff:4*ncoeff]

    # Convert to grid space
    up = expand(upbasis, np.reshape(upcoeff, (Lmax,Nmax)))
    um = expand(umbasis, np.reshape(umcoeff, (Lmax,Nmax)))
    w  = expand(u0basis, np.reshape( wcoeff, (Lmax,Nmax)))
    p  = expand( pbasis, np.reshape( pcoeff, (Lmax,Nmax)))
    u =       np.sqrt(0.5) * (up + um)
    v = -1j * np.sqrt(0.5) * (up - um)

    ns, neta = len(s), len(eta)
    ss, ee = s.reshape(1,ns), eta.reshape(neta,1)
    ur = np.sqrt(0.5) * ss * (up + um) + ee * np.sqrt(1-ss**2) * w

    return u, v, w, p, ur


def plot_solution(m, Lmax, Nmax, boundary_method, plot_evalues, plot_slices, plot_fields):
    save_plots = True
    plot_field_indices = [2,3,4]
#    mode_index = (4,2,1)
    mode_index = (60,29,1)

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

        plotfield(s, eta, f)
        plt.title(r'${}$'.format(field_names[field_index]))
        filename = prefix + '-evector-' + configstr + '-' + modestr + '-' + field_names[field_index] + '.png'
        savefig(filename)


def main():
    solve = True
    plot_evalues = True
    plot_fields = True
    plot_slices = False

    m = 1
    Lmax, Nmax = 10, 32
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

