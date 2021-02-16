import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import multiprocessing as mp

from dedalus_sphere import jacobi as Jacobi
from dedalus_sphere import operators
from .config import internal_dtype, max_processes


class Basis():
    def __init__(self, s, eta, m, Lmax, Nmax, sigma=0, alpha=0, beta=0, galerkin=False, truncate=False, dtype='float64', internal='float128', lazy=True, parallel=False):
        self.s, self.eta = s, eta
        self.m, self.Lmax, self.Nmax = m, Lmax, Nmax
        self.sigma, self.alpha, self.beta = sigma, alpha, beta
        self.dtype, self.internal = dtype, internal
        self.galerkin = galerkin
        self.parallel = parallel
        self.truncate = truncate

        if truncate:
            _check_radial_degree(Lmax, Nmax)

        if not lazy:
            self._construct_basis()
        else:
            self.sbasis, self.etabasis = None, None
            self._constructed = False

    @property
    def ncoeffs(self):
        Lmax, Nmax = self.Lmax, self.Nmax
        if self.truncate:
            return np.sum([Nmax-ell//2 for ell in range(Lmax)])
        else:
            return Lmax*Nmax

    class _Constructor():
        """Helper object to parallelize basis construction"""
        def __init__(self, Nmax, m, sigma, alpha, beta, t, onept, onemt, truncate, internal, dtype):
            self.Nmax, self.m, self.sigma, self.alpha, self.beta = Nmax, m, sigma, alpha, beta
            self.truncate, self.internal, self.dtype = truncate, internal, dtype
            self.t, self.onept, self.onemt = t, onept, onemt
        def __call__(self, ell):
            m, sigma, alpha, beta = self.m, self.sigma, self.alpha, self.beta
            internal, dtype = self.internal, self.dtype
            t, onept, onemt = self.t, self.onept, self.onemt
            N = self.Nmax - (ell//2 if self.truncate else 0)
            return ((onept * onemt**ell) * Jacobi.polynomials(N,ell+alpha-beta+1/2,m+sigma,t,dtype=internal)).astype(dtype)


    def _construct_basis(self):
        m, Lmax, Nmax = self.m, self.Lmax, self.Nmax
        sigma, alpha, beta = self.sigma, self.alpha, self.beta
        truncate, dtype, internal = self.truncate, self.dtype, self.internal

        s, eta = self.s.astype(internal), self.eta.astype(internal)
        t = 2*s**2 - 1
        if self.galerkin:
            escale, sscale = 1-eta**2, 1-s**2
        else:
            escale, sscale = 1, 1

        # Construct the eta basis
        etabasis = (escale * Jacobi.polynomials(Lmax,alpha,alpha,eta,dtype=internal)).T
        etabasis = etabasis.astype(dtype)

        # Construct the s basis
        onept = sscale * (1+t)**((m+sigma)/2)
        onemt = (1-t)**(1/2)
        fun = Basis._Constructor(Nmax, m, sigma, alpha, beta, t, onept, onemt, truncate, internal, dtype)
        if self.parallel:
            num_processes = min(mp.cpu_count(), max_processes)
            pool = mp.Pool(num_processes)
            sbasis = pool.map(fun, range(Lmax))
        else:
            sbasis = [fun(ell) for ell in range(Lmax)]

        self.sbasis, self.etabasis = sbasis, etabasis
        self._constructed = True


    def expand(self, coeffs):
        if not self._constructed:
            self._construct_basis()

        if np.prod(np.shape(coeffs)) != self.ncoeffs:
            raise ValueError('Inconsistent size')

        # Flatten the coefficients
        coeffs = coeffs.ravel()

        # Expand
        f = np.zeros((len(self.eta), len(self.s)), dtype=coeffs.dtype)
        Peta = self.etabasis
        index = 0
        for ell in range(self.Lmax):
            N = self.Nmax - (ell//2 if self.truncate else 0)
            Ps = self.sbasis[ell]
            f += Peta[:,ell][:,np.newaxis] * (coeffs[index:index+N] @ Ps)
            index += N
        return f


    def __getitem__(self, index):
        ell, k = index[0], index[1]
        if ell >= self.Lmax:
            raise ValueError('ell index out of range')
        N = self.Nmax - (ell//2 if self.truncate else 0)
        if k >= N:
            raise ValueError('k index out of range')

        if not self._constructed:
            self._construct_basis()
        return self.etabasis[:,ell][:,np.newaxis] * self.sbasis[ell][k,:]


def plotfield(s, eta, f, fig=None, ax=None, stretch=False, aspect='equal', colorbar=True):
    """Plot a 2D slice of the field at phi = 0"""
    s, eta = s.ravel(), eta.ravel()
    ss = s[np.newaxis,:]
    ee = eta[:,np.newaxis]
    if stretch:
        y = ee
    else:
        y = np.sqrt(1-ss**2)*ee
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4.25,6))
    im = ax.pcolormesh(ss, y, f, cmap='RdBu', shading='gouraud')
    if colorbar:
        fig.colorbar(im, ax=ax)
    ax.set_xlabel('s')
    ax.set_ylabel('η' if stretch else 'z')
    if aspect is not None:
        ax.set_aspect(aspect, adjustable='box')
    fig.set_tight_layout(True)


def resize(mat, Lin, Nin, Lout, Nout):
    """Reshape the matrix from codomain size (Lin,Nin) to size (Lout,Nout).
       This appends and deletes rows as necessary without touching the columns.
       Nin and Nout are functions of ell and return the number of radial coefficients
       for each vertical degree"""
    if np.isscalar(Nin):
        ninvalue = Nin
        Nin = lambda _: ninvalue
    nintotal = sum([Nin(ell) for ell in range(Lin)])

    if np.isscalar(Nout):
        noutvalue = Nout
        Nout = lambda _: noutvalue
    nouttotal = sum([Nout(ell) for ell in range(Lout)])

    nrows, ncols = np.shape(mat)
    if not nintotal == nrows:
        raise ValueError('Incorrect size')

    if not isinstance(mat, sparse.csr_matrix):
        mat = mat.tocsr()
    rows, cols = mat.nonzero()

    # If we have the zero matrix just return a zero matrix
    if len(rows) == 0:
        return sparse.lil_matrix((nouttotal, ncols))

    oprows, opcols, opdata = [], [], []
    L = min(Lin,Lout)
    inoffset, dn = 0, 0
    for ell in range(L):
        nin, nout = Nin(ell), Nout(ell)
        n = min(nin,nout)

        indices = np.where(np.logical_and(inoffset <= rows, rows < inoffset+n))
        if len(indices[0]) != 0:
            r, c = rows[indices], cols[indices]
            oprows += (r+dn).tolist()
            opcols += c.tolist()
            opdata += np.asarray(mat[r,c]).ravel().tolist()

        dn += nout-nin
        inoffset += nin

    result = sparse.csr_matrix((opdata,(oprows,opcols)), shape=(nouttotal,ncols), dtype=mat.dtype)

    return result


def triangular_truncate(mat, Lvar, Nvar, Leqn=None, Neqn=None):
    if Leqn is None: Leqn = Lvar
    if Neqn is None: Neqn = Nvar

    _check_radial_degree(Lvar, Nvar)
    _check_radial_degree(Leqn, Neqn)

    # truncate the equation space (output space)
    mat = resize(mat, Leqn, Neqn, Leqn, lambda ell: Neqn-ell//2)

    # truncate the variable space (input space)
    mat = resize(mat.T, Lvar, Nvar, Lvar, lambda ell: Nvar-ell//2).T

    return mat


def remove_zero_rows(mat):
    rows, cols = mat.nonzero()
    zrows = list(set(range(np.shape(mat)[0])) - set(rows))
    if not zrows:
        return mat
    for z in zrows:
        i = np.argmax(rows > z)
        if i > 0:
            rows[i:] -= 1
    return sparse.csr_matrix((mat.data, (rows,cols)), shape=(max(rows)+1,np.shape(mat)[1]))


def _check_radial_degree(Lmax, Nmax):
    if Nmax < Lmax//2:
        raise ValueError('Radial degree too small for triangular truncation')


def _hstack(*args, **kwargs):
    return sparse.hstack(*args, **kwargs, format='csr')


def _widest_dtype(zmat, smats):
    z = np.zeros(1,dtype=zmat.dtype)
    for smat in smats:
        if np.isscalar(smat):
            z = z+0*smat
        else:
            z = z+np.zeros(1,dtype=smat.dtype)
    return z.dtype


def make_operator(zmat, smats, Lmax=None, Nmax=None):
    """Kronecker out an operator.  Since the radial operators depend on ell,
       we require a separate operator matrix for each vertical expansion coefficient."""
    def pad(mat,n):
        if np.isscalar(mat):       return mat
        if np.shape(mat) == (1,1): return mat
        mat = mat[:n,:]
        nrows, ncols = np.shape(mat)
        if nrows < n:
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
    oprows, opcols, opdata = [], [], []
    rows, cols = zmat.nonzero()
    for row, col in zip(rows, cols):
        value = zmat[row,col]
        if not np.isscalar(value):
            value = value.todense()[0,0]
        if np.abs(value) < 1e-15:
            continue

        mat = sparse.csr_matrix(value * smats[col])
        matrows, matcols = mat.nonzero()
        oprows += (Nout*row + matrows).tolist()
        opcols += (Nin*col + matcols).tolist()
        opdata += np.asarray(mat[matrows,matcols]).ravel().tolist()

    dtype = _widest_dtype(zmat, smats)
    op = sparse.csr_matrix((opdata, (oprows, opcols)), shape=(Lout*Nout,Lin*Nin), dtype=dtype)
    return op


class Codomain():
    def __init__(self, dell, dn, dalpha):
        self._arrow = (dell,dn,dalpha)

    @property
    def arrow(self):
        return self._arrow

    def __getitem__(self, item):
        return self._arrow[item]

    def __call__(self,*args):
        return tuple(a+b for a,b in zip(self.arrow, args))

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = f'(ell->ell+{self[0]},n->n+{self[1]},α->α+{self[2]})'
        return s.replace('+0','').replace('+-','-')

    def __eq__(self, other):
        """Compare the numerical index α"""
        return self[2:] == other[2:]
    
    def __add__(self, other):
        if self != other:
            raise TypeError('operators have incompatible codomains.')
        return Codomain(*tuple(max(a,b) for a,b in zip(self[:2],other[:2])), *self[2:])

    def __mul__(self, other):
        return Codomain(*tuple(a+b for a,b in zip(self[:],other[:])))


class Operator():
    def __init__(self, codomain, dtype, internal):
        self._codomain = codomain

        if (np.zeros(1,dtype=dtype) + np.zeros(1,dtype=internal)).dtype == np.zeros(1,dtype=dtype).dtype:
            # Promote internal to the widest type
            internal = dtype
        self.dtype = dtype
        self.internal = internal

        self.A = Jacobi.operator('A', dtype=internal)
        self.B = Jacobi.operator('B', dtype=internal)
        self.C = Jacobi.operator('C', dtype=internal)
        self.D = Jacobi.operator('D', dtype=internal)
        self.Z = Jacobi.operator('Z', dtype=internal)
        self.Id = Jacobi.operator('Id', dtype=internal)

    @property
    def codomain(self):
        return self._codomain


class Boundary(Operator):
    """Evaluate a field on the ball boundary"""
    def __init__(self, dtype='float64', internal=internal_dtype):
        Operator.__init__(self, codomain=None, dtype=dtype, internal=internal)

    def codomain(self, m, Lmax, Nmax, alpha, sigma):
        L = Lmax-1
        return (Nmax+L//2, L//2+alpha+1/2, m+sigma), (Nmax+(L-1)//2, (L+1)//2+alpha+1/2, m+sigma)

    def __call__(self, m, Lmax, Nmax, alpha, sigma, truncate=False, separate=False):
        if truncate:
            _check_radial_degree(Lmax, Nmax)
        A = self.A
        L = Lmax-1

        bc = Jacobi.polynomials(Lmax,alpha,alpha,1.,dtype=self.internal)

        nrows = [Nmax, Nmax] if truncate else [Nmax+L//2, Nmax+(L-1)//2]
        ncols = np.sum([Nmax-ell//2 for ell in range(Lmax)]) if truncate else Lmax*Nmax
        matrix = lambda shape: sparse.lil_matrix(shape, dtype=self.internal)
        Opeven, Opodd = tuple(matrix((nr,ncols)) for nr in nrows)
        index = 0
        for ell in range(Lmax):
            N = Nmax - (ell//2 if truncate else 0)
            op = bc[ell] * ((A(+1)**((L-ell)//2)) @ (A(-1)**(ell//2)))(N, ell+alpha+1/2, m+sigma)
            mat = [Opeven, Opodd][ell % 2]
            mat[:np.shape(op)[0],index:index+N] = op
            index += N

        Opeven, Opodd = Opeven.astype(self.dtype), Opodd.astype(self.dtype)
        if separate:
            return Opeven, Opodd
        else:
            return sparse.vstack([Opeven,Opodd], format='csr')


class Conversion(Operator):
    """Convert up in alpha index.  This isn't really a tensor operation since it can
       act independently on components of vectors"""
    def __init__(self, dtype='float64', internal=internal_dtype):
        Operator.__init__(self, codomain=Codomain(0,+1,+1), dtype=dtype, internal=internal)

    def __call__(self, m, Lmax, Nmax, alpha, sigma):
        A, B = self.A, self.B
        opz = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()                   # (ell,alpha,alpha) -> (ell,alpha+1,alpha+1)
        gamma_ell = np.diag(opz)
        delta_ell = np.diag(opz,2)
 
        zmat = np.diag(gamma_ell)
        smats = [A(+1)(Nmax,ell+alpha+1/2,m+sigma) for ell in range(Lmax)]  # (n,a,b) -> (n,a+1,b)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
 
        zmat = np.diag(delta_ell,2)
        smats = [A(-1)(Nmax,ell+alpha+1/2,m+sigma) for ell in range(Lmax)]  # (n,a,b) -> (n+1,a-1,b)
        Op2 = make_operator(zmat, smats)
 
        Op = Op1 + Op2
        return Op.astype(self.dtype)


class RadialVector(Operator):
    """Extract the spherical radial part of a velocity field"""   
    def __init__(self, dtype='float64', internal=internal_dtype):
        codomain = [Codomain(0,+1,0), Codomain(0,0,0), Codomain(+1,+1,0)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal)

    def __call__(self, m, Lmax, Nmax, alpha):
        A, B, Z, Id = self.A, self.B, self.Z, self.Id

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
        return _hstack([Opp, Opm, Opz]).astype(self.dtype)


class RadialMultiplication(Operator):
    """Multiply a scalar field by the spherical radius vector"""
    def __init__(self, dtype='float64', internal=internal_dtype, convert=True):
        codomain = [Codomain(0,0,0), Codomain(0,+1,0), Codomain(+1,+1,0)] 
        if convert:
            codomain = [Conversion().codomain * cd for cd in codomain]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal)
        self.convert = convert

    def __call__(self, m, Lmax, Nmax, alpha):
        A, B, Z, Id = self.A, self.B, self.Z, self.Id

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

        if self.convert:
            conversion = Conversion(dtype=self.internal, internal=self.internal)
            Opp = conversion(m, Lmax,   Nmax  , alpha=alpha, sigma=+1) @ Opp
            Opm = conversion(m, Lmax,   Nmax+1, alpha=alpha, sigma=-1) @ Opm
            Opz = conversion(m, Lmax+1, Nmax+1, alpha=alpha, sigma= 0) @ Opz

        return Opp.astype(self.dtype), Opm.astype(self.dtype), Opz.astype(self.dtype)


class OneMinusRadiusSquared(Operator):
    """Multiply a field by (1-r**2)"""
    def __init__(self, dtype='float64', internal=internal_dtype):
        codomain = Codomain(+2,+1,-1)
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal)

    def __call__(self, m, Lmax, Nmax, alpha, sigma):
        A, B = self.A, self.B

        opz = (A(-1) @ B(-1))(Lmax,alpha,alpha).todense()
        zmat = np.diag(np.diag(opz))
        smats = [A(-1)(Nmax,ell+alpha+1/2,m+sigma) for ell in range(Lmax)]    # (n,a,b) -> (n+1,a-1,b)
        Op1 = make_operator(zmat, smats, Lmax=Lmax+2)

        zmat = np.diag(np.diag(opz,-2),-2)[:,:Lmax]
        smats = [A(+1)(Nmax,ell+alpha+1/2,m+sigma) for ell in range(Lmax)]    # (n,a,b) -> (n,a+1,b)
        Op2 = make_operator(zmat, smats, Nmax=Nmax+1)

        Op = 1/2*(Op1 + Op2)

        return Op.astype(self.dtype)


class RdR(Operator):
    """r d/dr operator on a field"""
    def __init__(self, dtype='float64', internal=internal_dtype):
        Operator.__init__(self, codomain=Codomain(0,+1,+1), dtype=dtype, internal=internal)

    def __call__(self, m, Lmax, Nmax, alpha, sigma):
        A, B, C, D = self.A, self.B, self.C, self.D

        op = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()
        gamma_ell, delta_ell = np.diag(op), -np.diag(op,2)

        zmat = np.diag(gamma_ell)
        smats = [((ell-(m+sigma))*A(+1) + 2*(B(+1) @ C(+1)))(Nmax,ell+alpha+1/2,m+sigma) for ell in range(Lmax)]
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)

        zmat = np.diag(delta_ell,2)
        smats = [((ell+2*alpha+1+m+sigma)*A(-1) + 2*(B(+1) @ D(-1)))(Nmax,ell+alpha+1/2,m+sigma) for ell in range(Lmax)]
        Op2 = make_operator(zmat, smats)

        Op = Op1 + Op2

        return Op.astype(self.dtype)


class Gradient(Operator):
    """Compute the gradient of a scalar field"""
    def __init__(self, dtype='float64', internal=internal_dtype):
        codomain = [Codomain(0,0,+1), Codomain(0,+1,+1), Codomain(-1,0,+1)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal)

    def __call__(self, m, Lmax, Nmax, alpha):
        A, B, C, D, Id = self.A, self.B, self.C, self.D, self.Id

        op = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()
        gamma_ell = np.diag(op)
        delta_ell = -np.diag(op,2)
    
        # e(+)^* . Grad
        zmat = np.diag(2*gamma_ell)
        smats = [D(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]  # (n,a,b) -> (n-1,a+1,b+1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax)
        zmat = np.diag(2*delta_ell,2)
        smats = [C(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]  # (n,a,b) -> (n,a-1,b+1)
        Op2 = make_operator(zmat, smats)
        Opp = Op1 + Op2
    
        # e(-)^* . Grad
        zmat = np.diag(2*gamma_ell)
        smats = [C(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]  # (n,a,b) -> (n,a+1,b-1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
        zmat = np.diag(2*delta_ell,2)
        smats = [D(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]  # (n,a,b) -> (n+1,a-1,b-1)
        Op2 = make_operator(zmat, smats)
        Opm = Op1 + Op2

        # e(z)^* . Grad
        zmat = np.sqrt(2) * D(+1)(Lmax,alpha,alpha)                   # (ell,alpha,alpha) -> (ell-1,alpha+1,alpha+1)
        smats = [Id(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]
        Opz = make_operator(zmat, smats)

        return Opp.astype(self.dtype), Opm.astype(self.dtype), Opz.astype(self.dtype)


class Divergence(Operator):
    """Compute the divergence of a vector field"""
    def __init__(self, dtype='float64', internal=internal_dtype):
        codomain = [Codomain(0,+1,+1), Codomain(0,0,+1), Codomain(-1,0,+1)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal)
     
    def __call__(self, m, Lmax, Nmax, alpha):
        A, B, C, D, Id = self.A, self.B, self.C, self.D, self.Id

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

        return _hstack([Opp, Opm, Opz]).astype(self.dtype)


class Curl(Operator):
    """Compute the divergence of a vector field"""
    def __init__(self, dtype='float64', internal=internal_dtype):
        codomain = [Codomain(0,0,+1), Codomain(0,+1,+1), Codomain(0,+1,+1)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal)
     
    def __call__(self, m, Lmax, Nmax, alpha):
        A, B, C, D, Id = self.A, self.B, self.C, self.D, self.Id

        op = (A(+1) @ B(+1))(Lmax,alpha,alpha).todense()                # (ell,alpha,alpha) -> (ell,alpha+1,alpha+1)
        gamma_ell = np.diag(op)
        delta_ell = -np.diag(op,2)

        # e(+)^* . Curl
        zmat = 1 * np.sqrt(2) * D(+1)(Lmax,alpha,alpha)                # (ell,alpha,alpha) -> (ell-1,alpha+1,alpha+1)
        smats = [Id(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]
        Opp_p = make_operator(zmat, smats, Lmax=Lmax)

        zmat = -2 * np.diag(gamma_ell)
        smats = [D(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]    # (n,a,b) -> (n-1,a+1,b+1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax)
        zmat = -2 * np.diag(delta_ell,2)
        smats = [C(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]    # (n,a,b) -> (n,a-1,b+1)
        Op2 = make_operator(zmat, smats)
        Opp_z = Op1 + Op2

        # e(-)^* . Curl
        zmat = -1 * np.sqrt(2) * D(+1)(Lmax,alpha,alpha)               # (ell,alpha,alpha) -> (ell-1,alpha+1,alpha+1)
        smats = [Id(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]
        Opm_m = make_operator(zmat, smats, Lmax=Lmax, Nmax=Nmax+1)

        zmat = 2 * np.diag(gamma_ell)
        smats = [C(+1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]    # (n,a,b) -> (n,a+1,b-1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
        zmat = 2 * np.diag(delta_ell,2)
        smats = [D(-1)(Nmax,ell+alpha+1/2,m) for ell in range(Lmax)]    # (n,a,b) -> (n+1,a-1,b-1)
        Op2 = make_operator(zmat, smats)
        Opm_z = Op1 + Op2

        # e(z)^* . Curl
        zmat = -2 * np.diag(gamma_ell)
        smats = [C(+1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]  # (n,a,b) -> (n,a+1,b-1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
        zmat = -2 * np.diag(delta_ell,2)
        smats = [D(-1)(Nmax,ell+alpha+1/2,m+1) for ell in range(Lmax)]  # (n,a,b) -> (n+1,a-1,b-1)
        Op2 = make_operator(zmat, smats)
        Opz_p = Op1 + Op2

        zmat = 2 * np.diag(gamma_ell)
        smats = [D(+1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]  # (n,a,b) -> (n-1,a+1,b+1)
        Op1 = make_operator(zmat, smats, Nmax=Nmax+1)
        zmat = 2 * np.diag(delta_ell,2)
        smats = [C(-1)(Nmax,ell+alpha+1/2,m-1) for ell in range(Lmax)]  # (n,a,b) -> (n,a-1,b+1)
        Op2 = make_operator(zmat, smats, Nmax=Nmax+1)
        Opz_m = Op1 + Op2

        Zp, Zm, Zz = 0*Opp_p, 0*Opm_m, 0*Opz_p
        return 1j*(_hstack([Opp_p, Zp, Opp_z]).astype(self.dtype)), \
               1j*(_hstack([Zm, Opm_m, Opm_z]).astype(self.dtype)), \
               1j*(_hstack([Opz_p, Opz_m, Zz]).astype(self.dtype))


class ScalarLaplacian(Operator):
    def __init__(self, dtype='float64', internal=internal_dtype):
        Operator.__init__(self, codomain=Codomain(0,+1,+2), dtype=dtype, internal=internal)

    def __call__(self, m, Lmax, Nmax, alpha):
        kwargs = {'dtype':self.internal, 'internal':self.internal}
        divergence, gradient = Divergence(**kwargs), Gradient(**kwargs)

        gradp, gradm, gradz = gradient(m, Lmax, Nmax, alpha)
        gradp = resize(gradp, Lmax, Nmax, Lmax, Nmax+1)
        gradz = resize(gradz, Lmax-1, Nmax, Lmax, Nmax+1)
        grad = sparse.vstack([gradp,gradm,gradz])
        div = divergence(m, Lmax, Nmax+1, alpha+1)
        dg = div @ grad
        dg = resize(dg, Lmax, Nmax+2, Lmax, Nmax+1)
        return dg.astype(self.dtype)


class VectorLaplacian(Operator):
    def __init__(self, dtype='float64', internal=internal_dtype):
        codomain = [Codomain(0,+1,+2), Codomain(0,+1,+2), Codomain(0,+1,+2)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal)

    def __call__(self, m, Lmax, Nmax, alpha):
        kwargs = {'dtype':self.internal, 'internal':self.internal}
        divergence, gradient, curl = Divergence(**kwargs), Gradient(**kwargs), Curl(**kwargs)

        # Curl(Curl)
        curlp, curlm, curlz = curl(m, Lmax, Nmax, alpha)
        curlp = resize(curlp, Lmax, Nmax, Lmax, Nmax+1)
        curl1 = sparse.vstack([curlp,curlm,curlz])
        curlp, curlm, curlz = curl(m, Lmax, Nmax+1, alpha+1)
        curlp = resize(curlp, Lmax, Nmax+1, Lmax, Nmax+2)
        curl2 = sparse.vstack([curlp,curlm,curlz])
        cc = (curl2 @ curl1).real

        # Grad(Div)
        div = divergence(m, Lmax, Nmax, alpha)
        gradp, gradm, gradz = gradient(m, Lmax, Nmax+1, alpha+1)
        gradp = resize(gradp, Lmax, Nmax+1, Lmax, Nmax+2)
        gradz = resize(gradz, Lmax-1, Nmax+1, Lmax, Nmax+2)
        grad = sparse.vstack([gradp,gradm,gradz])
        gd = grad @ div

        # Vector Laplacian
        op = gd - cc
        rows, cols, _ = sparse.find(abs(op) >= 1e-12)
        values = [op[r,c] for r,c in zip(rows, cols)]
        op = sparse.csr_matrix((values,(rows,cols)), shape=np.shape(op)).astype(self.dtype)

        nin, nout = Lmax*Nmax, Lmax*(Nmax+2)
        Opp, Opm, Opz = op[:nout,:nin], op[nout:2*nout,nin:2*nin], op[2*nout:,2*nin:]
        Opp = resize(Opp, Lmax, Nmax+2, Lmax, Nmax+1)
        Opm = resize(Opm, Lmax, Nmax+2, Lmax, Nmax+1)
        Opz = resize(Opz, Lmax, Nmax+2, Lmax, Nmax+1)
        return Opp, Opm, Opz


def operator(name, field=None, dtype='float64', internal=internal_dtype):
    if name in ['divergence', 'div']:
        return Divergence(dtype=dtype, internal=internal)
    if name in ['gradient', 'grad']:
        return Gradient(dtype=dtype, internal=internal)
    if name == 'curl':
        return Curl(dtype=dtype, internal=internal)
    if name in ['laplacian', 'lap']:
        if field in ['vector', 'vec']:
            op = VectorLaplacian(dtype=dtype, internal=internal)
        else:
            op = ScalarLaplacian(dtype=dtype, internal=internal)
        return op
    if name == 'rtimes':
        return RadialMultiplication(dtype=dtype, internal=internal)
    if name == '1-r**2':
        return OneMinusRadiusSquared(dtype=dtype, internal=internal)
    if name == 'rdot':
        return RadialVector(dtype=dtype, internal=internal)
    if name == 'r*d/dr':
        return RdR(dtype=dtype, internal=internal)
    if name in ['boundary', 'r=1']:
        return Boundary(dtype=dtype, internal=internal)
    if name == 'conversion':
        return Conversion(dtype=dtype, internal=internal)

    raise ValueError('Unknown operator')


def convert_alpha(ntimes, m, Lmax, Nmax, alpha, sigma, truncate, dtype='float64', internal=internal_dtype):
    Conv = operator('conversion', dtype=internal, internal=internal)
    op = sparse.eye(Lmax*Nmax, format='csr', dtype=internal)
    for i in range(ntimes):
        op1 = Conv(m, Lmax, Nmax+i, alpha=alpha+i, sigma=sigma)
        op = op1 @ op
    Ntrunc = Nmax if truncate else Nmax+1
    op = resize(op, Lmax, Nmax+ntimes, Lmax, Ntrunc)
    return op.astype(dtype)
    

def convert_beta(m, Lmax, Nmax, alpha, sigma, beta, dtype='float64', internal=internal_dtype):
    A = Jacobi.operator('A', dtype=internal)

    zmat = sparse.eye(Lmax)
    smats = [(A(+1)**beta)(Nmax,ell+alpha-beta+1/2,m+sigma) for ell in range(Lmax)]
    op = make_operator(zmat, smats)
    return op.astype(dtype)

