import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from dedalus_sphere import jacobi as Jacobi

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
    """Plot a 2D slice of the field at phi = 0"""
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


def alpha_conversion(m, Lmax, Nmax, alpha, sigma, Ntrunc=None):
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


def spherical_radial_vector(m, Lmax, Nmax, alpha, Ntrunc=None):
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

    return Opp, Opm, Opz


def r_multiplication(m, Lmax, Nmax, alpha, Ntrunc=None):
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


def gradient(m, Lmax, Nmax, alpha, Ntrunc=None):
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

    return Opp, Opm, Opz


def divergence(m, Lmax, Nmax, alpha, Ntrunc=None):
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

    return Opp, Opm, Opz


def curl(m, Lmax, Nmax, alpha, Ntrunc=None):
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


def scalar_laplacian(m, Lmax, Nmax, alpha):
    Grad = make_gradient_operator(m, Lmax, Nmax, alpha, Ntrunc=Nmax+1)
    Div = make_divergence_operator(m, Lmax, Nmax+1, alpha+1, Ntrunc=Nmax)
    return Div @ Grad


def vector_laplacian(m, Lmax, Nmax, alpha, incompressible=False):
    Curl1 = make_curl_operator(m, Lmax, Nmax, alpha, Ntrunc=Nmax+1)
    Curl2 = make_curl_operator(m, Lmax, Nmax+1, alpha+1, Ntrunc=Nmax)
    Lap = -Curl2 @ Curl1
    Lap = Lap.real

    if not incompressible:
        Div = make_divergence_operator(m, Lmax, Nmax, alpha, Ntrunc=Nmax+1)
        Grad = make_gradient_operator(m, Lmax, Nmax+1, alpha+1, Ntrunc=Nmax)
        Lap += Grad @ Div

    return Lap


def _boundary_evaluation_impl(m, Lmax, Nmax, alpha, sigma):
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


def boundary_evaluation(m, Lmax, Nmax, alpha, sigma):
    make_op = lambda s: _boundary_evaluation_impl(m, Lmax, Nmax, alpha, s)
    if np.isscalar(sigma):
        return make_op(sigma)
    else:
        return sparse.hstack([make_op(s) for s in sigma])


