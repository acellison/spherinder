import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from dedalus_sphere import jacobi as Jacobi
from dedalus_sphere import operators


def psi(Nmax, m, ell, s, eta, sigma=0, alpha=0):
    """Basis function for our fields"""
    ns, neta = len(s), len(eta)
    t = 2*s**2 - 1

    Peta = Jacobi.polynomials(ell+1,alpha,alpha,eta)[-1,:].reshape(neta,1)
    Ps = Jacobi.polynomials(Nmax,ell+alpha+1/2,m+sigma,t)
    tt = t.reshape(1,ns)
    return [Peta * (1+tt)**((m+sigma)/2) * (1-tt)**(ell/2) * Ps[k,:] for k in range(Nmax)]


def expand(basis, coeffs):
    """Expand the coefficient vector to grid space"""
    f = np.zeros(np.shape(basis[0][0]), dtype=coeffs.dtype)
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


def hstack(*args, **kwargs):
    return sparse.hstack(*args, **kwargs, format='csr')


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

    zmat = operators.infinite_csr(zmat)
    Lout, Lin = np.shape(zmat)
    Nout, Nin = 0, 0
    for smat in smats:
        if np.ndim(smat) < 2: continue
        sh = np.shape(smat)
        if Nout < sh[0]: Nout = sh[0]
        if Nin  < sh[1]: Nin  = sh[1]

    if Lmax is not None: Lout = Lmax
    if Nmax is not None: Nout = Nmax
    zmat = pad(zmat, Lout)
    smats = [pad(smat, Nout) for smat in smats]

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


class Codomain(operators.Codomain):
    def __init__(self, dell, dn, dsigma, dalpha):
        operators.Codomain.__init__(self, *(dell,dn,dsigma,dalpha), Output=Codomain)

    def __str__(self):
        s = f'(ell->ell+{self[0]},n->n+{self[1]},σ->σ+{self[2]},α->α+{self[3]})'
        return s.replace('+0','').replace('+-','-')

    def __eq__(self, other):
        return self[2:] == other[2:]
    
    def __or__(self, other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        return Codomain(*tuple(max(a,b) for a,b in zip(self[:2],other[:2])), *self[2:])


class Operator():
    def __init__(self, codomain):
        self._codomain = codomain


    @property
    def codomain(self):
        return self._codomain


class Conversion(Operator):
    """Convert up in alpha index"""
    def __init__(self):
        Operator.__init__(self, Codomain(0,+1,0,+1))


    def __call__(self, m, Lmax, Nmax, sigma, alpha):
        opz = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()                   # (ell,alpha,alpha) -> (ell,alpha+1,alpha+1)
        alpha_ell = np.diag(opz)
        beta_ell = np.diag(opz,2)
 
        zmat = np.diag(alpha_ell)
        smats = [A(+1)(Nmax,ell+alpha+1/2,m+sigma) for ell in range(Lmax)]  # (n,a,b) -> (n,a+1,b)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
 
        zmat = np.diag(beta_ell,2)
        smats = [A(-1)(Nmax,ell+alpha+1/2,m+sigma) for ell in range(Lmax)]  # (n,a,b) -> (n+1,a-1,b)
        Op2 = make_operator(zmat, smats)
 
        Op = Op1 + Op2
        return Op


class RadialVector(Operator):
    """Extract the spherical radial part of a velocity field"""   
    def __init__(self):
        Operator.__init__(self, Codomain(+1,+1,0,0))


    def __call__(self, m, Lmax, Nmax, alpha):
        # Coeff space operator: s * u(+)
        zmat = Id(Lmax,alpha,alpha)
        smats = [B(-1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]          # (n,a,b) -> (n+1,a,b-1)
        Opp = 1/2 * make_operator(zmat, smats, Nmax=Nmax+1, Lmax=Lmax+1)
 
        # Coeff space operator: s * u(-)
        zmat = Id(Lmax,alpha,alpha)
        smats = [B(+1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]          # (n,a,b) -> (n,a,b+1)
        Opm = 1/2 * make_operator(zmat, smats, Nmax=Nmax+1, Lmax=Lmax+1)
 
        # Coeff space operator: z * w = eta * (1-s**2)**0.5 * w
        opz = Z(Lmax,alpha,alpha).todense()                                     # (ell,alpha,alpha) -> (ell+1,alpha,alpha)
 
        zmat = np.diag(np.diag(opz,-1),-1)[:,:Lmax]
        smats = [A(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)] + [0]      # (n,a,b) -> (n,a+1,b)
        Opz1 = make_operator(zmat, smats, Nmax=Nmax+1)
 
        zmat = np.diag(np.diag(opz,+1),+1)
        smats = [0] + [A(-1)(Nmax,ell+alpha+1/2,m) for ell in range(1,Lmax+1)]  # (n,a,b) -> (n+1,a-1,b)
        Opz2 = make_operator(zmat, smats, Lmax=Lmax+1)

        Opz = 1/np.sqrt(2) * (Opz1 + Opz2)
        return hstack([Opp, Opm, Opz])


class RadialMultiplication(Operator):
    """Multiply a scalar field by the spherical radius vector"""
    def __init__(self):
        codomains = [Codomain(0,0,+1,0), Codomain(0,+1,-1,0), Codomain(+1,+1,0,0)]
        conversion = Conversion()
        Operator.__init__(self, tuple([[conversion.codomain + cd] for cd in codomains]))

    def __call__(self, m, Lmax, Nmax, alpha):
        # u(+) operator
        zmat = Id(Lmax,alpha,alpha)
        smats = [B(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]            # (n,a,b) -> (n,a,b+1)
        Opp = 1/2 * make_operator(zmat, smats)

        # u(-) operator
        zmat = Id(Lmax,alpha,alpha)
        smats = [B(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]            # (n,a,b) -> (n+1,a,b-1)
        Opm = 1/2 * make_operator(zmat, smats)

        # u(z) operator
        opz = Z(Lmax,alpha,alpha).todense()                                     # (ell,alpha,alpha) -> (ell+1,alpha,alpha)

        zmat = np.diag(np.diag(opz,-1),-1)[:,:Lmax]
        smats = [A(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax+1)]          # (n,a,b) -> (n,a+1,b)
        Opz1 = make_operator(zmat, smats, Nmax=Nmax+1)

        zmat = np.diag(np.diag(opz,+1),+1)
        smats = [0] + [A(-1)(Nmax,ell+alpha+1/2,m) for ell in range(1,Lmax+1)]  # (n,a,b) -> (n+1,a-1,b)
        Opz2 = make_operator(zmat, smats, Lmax=Lmax+1)

        Opz = 1/np.sqrt(2) * (Opz1 + Opz2)

        conversion = Conversion()
        Cp = conversion(m, Lmax,   Nmax,   sigma=+1, alpha=alpha)
        Cm = conversion(m, Lmax,   Nmax+1, sigma=-1, alpha=alpha)
        Cz = conversion(m, Lmax+1, Nmax+1, sigma= 0, alpha=alpha)

        return Cp @ Opp, Cm @ Opm, Cz @ Opz


class Gradient(Operator):
    """Compute the gradient of a scalar field"""
    def __init__(self):
        Operator.__init__(self, ([Codomain(0,0,+1,+1)], [Codomain(0,+1,-1,+1)], [Codomain(-1,0,0,+1)]))

    def __call__(self, m, Lmax, Nmax, alpha):
        op = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()
        alpha_ell = np.diag(op)
        beta_ell = -np.diag(op,2)
    
        # e(+)^* . Grad
        zmat = np.diag(2*alpha_ell)
        smats = [D(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]  # (n,a,b) -> (n-1,a+1,b+1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax)
        zmat = np.diag(2*beta_ell,2)
        smats = [C(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]  # (n,a,b) -> (n,a-1,b+1)
        Op2 = make_operator(zmat, smats)
        Opp = Op1 + Op2
    
        # e(-)^* . Grad
        zmat = np.diag(2*alpha_ell)
        smats = [C(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]  # (n,a,b) -> (n,a+1,b-1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
        zmat = np.diag(2*beta_ell,2)
        smats = [D(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]  # (n,a,b) -> (n+1,a-1,b-1)
        Op2 = make_operator(zmat, smats)
        Opm = Op1 + Op2

        # e(z)^* . Grad
        zmat = np.sqrt(2) * D(+1)(Lmax,alpha,alpha)                   # (ell,alpha,alpha) -> (ell-1,alpha+1,alpha+1)
        smats = [Id(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
        Opz = make_operator(zmat, smats)

        return Opp, Opm, Opz


class Divergence(Operator):
    """Compute the divergence of a vector field"""
    def __init__(self):
        Operator.__init__(self, Codomain(0,+1,0,+1))
     
    def __call__(self, m, Lmax, Nmax, alpha):
        op = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()                # (ell,alpha,alpha) -> (ell,alpha+1,alpha+1)
        gamma_ell = np.diag(op)
        delta_ell = -np.diag(op,2)
   
        # Div . e(+)^* .
        zmat = np.diag(2*gamma_ell)
        smats = [C(+1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]  # (n,a,b) -> (n,a+1,b-1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
        zmat = np.diag(2*delta_ell,2)
        smats = [D(-1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]  # (n,a,b) -> (n+1,a-1,b-1)
        Op2 = make_operator(zmat, smats)
        Opp = Op1 + Op2

        # Div . e(-)^* . 
        zmat = np.diag(2*gamma_ell)
        smats = [D(+1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]  # (n,a,b) -> (n-1,a+1,b+1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
        zmat = np.diag(2*delta_ell,2)
        smats = [C(-1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]  # (n,a,b) -> (n,a-1,b+1)
        Op2 = make_operator(zmat, smats, Nmax=Nmax+1)
        Opm = Op1 + Op2
 
        # Div . e(z)^* .
        zmat = np.sqrt(2) * D(+1)(Lmax,alpha,alpha)                     # (ell,alpha,alpha) -> (ell-1,alpha+1,alpha+1)
        smats = [Id(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
        Opz = make_operator(zmat, smats, Lmax=Lmax, Nmax=Nmax+1)

        return hstack([Opp, Opm, Opz])


class Curl(Operator):
    """Compute the divergence of a vector field"""
    def __init__(self):
        Operator.__init__(self, (Codomain(0,0,0,+1), Codomain(0,+1,0,+1), Codomain(0,+1,0,+1)))
     
    def __call__(self, m, Lmax, Nmax, alpha):
        op = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()                # (ell,alpha,alpha) -> (ell,alpha+1,alpha+1)
        gamma_ell = np.diag(op)
        delta_ell = -np.diag(op,2)

        # e(+)^* . Curl
        zmat = 1j * np.sqrt(2) * D(+1)(Lmax,alpha,alpha)                # (ell,alpha,alpha) -> (ell-1,alpha+1,alpha+1)
        smats = [Id(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]
        Opp_p = make_operator(zmat, smats, Lmax=Lmax)

        zmat = -2j * np.diag(gamma_ell)
        smats = [D(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]    # (n,a,b) -> (n-1,a+1,b+1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax)
        zmat = -2j * np.diag(delta_ell,2)
        smats = [C(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]    # (n,a,b) -> (n,a-1,b+1)
        Op2 = make_operator(zmat, smats)
        Opp_z = Op1 + Op2

        # e(-)^* . Curl
        zmat = -1j * np.sqrt(2) * D(+1)(Lmax,alpha,alpha)               # (ell,alpha,alpha) -> (ell-1,alpha+1,alpha+1)
        smats = [Id(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]
        Opm_m = make_operator(zmat, smats, Lmax=Lmax, Nmax=Nmax+1)

        zmat = 2j * np.diag(gamma_ell)
        smats = [C(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]    # (n,a,b) -> (n,a+1,b-1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
        zmat = 2j * np.diag(delta_ell,2)
        smats = [D(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]    # (n,a,b) -> (n+1,a-1,b-1)
        Op2 = make_operator(zmat, smats)
        Opm_z = Op1 + Op2

        # e(z)^* . Curl
        zmat = -2j * np.diag(gamma_ell)
        smats = [C(+1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]  # (n,a,b) -> (n,a+1,b-1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
        zmat = -2j * np.diag(delta_ell,2)
        smats = [D(-1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]  # (n,a,b) -> (n+1,a-1,b-1)
        Op2 = make_operator(zmat, smats)
        Opz_p = Op1 + Op2

        zmat = 2j * np.diag(gamma_ell)
        smats = [D(+1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]  # (n,a,b) -> (n-1,a+1,b+1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
        zmat = 2j * np.diag(delta_ell,2)
        smats = [C(-1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]  # (n,a,b) -> (n,a-1,b+1)
        Op2 = make_operator(zmat, smats, Nmax=Nmax+1)
        Opz_m = Op1 + Op2

        Zp, Zm, Zz = 0*Opp_p, 0*Opm_m, 0*Opz_p
        return hstack([Opp_p, Zp, Opp_z]), hstack([Zm, Opm_m, Opm_z]), hstack([Opz_p, Opz_m, Zz])


class Laplacian(Operator):
    """Compute the Laplacian of a field"""
    def __init__(self, field='scalar', incompressible=False):
        if field not in ['scalar', 'vector']:
            raise ValueError('Unsupported Laplacian field')
        self._field = field
        self._incompressible = incompressible

        if field == 'scalar':
            Operator.__init__(self, Codomain(0,+1,0,+2))
        else:
            # FIXME: implement
            pass

    @property
    def field(self):
        return self._field

    @property
    def incompressible(self):
        return self._incompressible

    def __call__(self, m, Lmax, Nmax, alpha):
        if self.field == 'scalar':
            return scalar_operator(m, Lmax, Nmax, alpha)
        else:
            return vector_operator(m, Lmax, Nmax, alpha)

    def scalar_operator(self, m, Lmax, Nmax, alpha):
        pass           
        
    def scalar_operator(self, m, Lmax, Nmax, alpha):
        pass


def reshape_codomain(mat, Lin, Nin, Lout, Nout):
    """Reshape the matrix from codomain size (Lin,Nin) to size (Lout,Nout).
       This appends and deletes rows as necessary without touching the input space (columns)"""
    nrows, ncols = np.shape(mat)
    if not Lin*Nin == nrows:
        raise ValueError('Incorrect size')

    result = sparse.lil_matrix((Lout*Nout,ncols), dtype=mat.dtype)
    L, N = min(Lin,Lout), min(Nin,Nout)
    for ell in range(L):
        result[ell*Nout:ell*Nout+N,:] = mat[ell*Nin:ell*Nin+N,:]
    return result        


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


def _boundary_evaluation_impl(m, Lmax, Nmax, sigma, alpha, separate=False):
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

    if separate:
        return Opeven, Opodd
    else:
        return sparse.vstack([Opeven,Opodd])


def boundary_codomain(m, Lmax, Nmax, sigma, alpha):
    """Return the codomain after collapsing the eta dependence through
       boundary evaluation.  Returns a tuple of (even codomain, odd codomain)"""
    L = Lmax-1
    return (Nmax+L//2, L//2+alpha+1/2, m+sigma), (Nmax+(L-1)//2, (L+1)//2+alpha+1/2, m+sigma)


def boundary_evaluation(m, Lmax, Nmax, sigma, alpha, separate=False):
    """Create the boundary evaluation operator.  separate returns a 
       pair of operators, one for even ell indices and one for odd ell indices.
       Can pass in the list [+1,-1,0] for sigma to form the vector field boundary
       evaluation operator"""
    make_op = lambda s: _boundary_evaluation_impl(m, Lmax, Nmax, s, alpha, separate=separate)
    if np.isscalar(sigma):
        return make_op(sigma)
    else:
        ops = [make_op(s) for s in sigma]
        if separate:
            return hstack([op[0] for op in ops]), sparse.hstack([op[1] for op in ops])
        else:
            return hstack(ops)


