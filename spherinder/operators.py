import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import multiprocessing as mp

from dedalus_sphere import jacobi as Jacobi
from dedalus_sphere import operators
from .config import internal_dtype, max_processes, default_truncate


class Basis():
    def __init__(self, s, eta, m, Lmax, Nmax, sigma=0, alpha=0, beta=0, galerkin=False, truncate=default_truncate, dtype='float64', internal='float128', lazy=True, parallel=False):
        self.s, self.eta = s, eta
        self.m, self.Lmax, self.Nmax = m, Lmax, Nmax
        self.sigma, self.alpha, self.beta = sigma, alpha, beta
        self.dtype, self.internal = dtype, internal
        self.galerkin = galerkin
        self.parallel = parallel
        self.truncate = truncate
        self.Nsizes = Nsizes(self.Lmax, self.Nmax, truncate=self.truncate)

        if not lazy:
            self._construct_basis()
        else:
            self.sbasis, self.etabasis = None, None
            self._constructed = False

    @property
    def ncoeffs(self):
        return num_coeffs(self.Lmax, self.Nmax, truncate=self.truncate)


    class _Constructor():
        """Helper object to parallelize basis construction"""
        def __init__(self, Nsizes, m, sigma, alpha, beta, t, onept, onemt, truncate, internal, dtype):
            self.Nsizes, self.m, self.sigma, self.alpha, self.beta = Nsizes, m, sigma, alpha, beta
            self.truncate, self.internal, self.dtype = truncate, internal, dtype
            self.t, self.onept, self.onemt = t, onept, onemt
        def __call__(self, ell):
            m, sigma, alpha, beta = self.m, self.sigma, self.alpha, self.beta
            internal, dtype = self.internal, self.dtype
            t, onept, onemt = self.t, self.onept, self.onemt
            N = self.Nsizes[ell]
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
        fun = Basis._Constructor(self.Nsizes, m, sigma, alpha, beta, t, onept, onemt, truncate, internal, dtype)
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
            N = self.Nsizes[ell]
            Ps = self.sbasis[ell]
            f += Peta[:,ell][:,np.newaxis] * (coeffs[index:index+N] @ Ps)
            index += N
        return f


    def __getitem__(self, index):
        ell, k = index[0], index[1]
        if ell >= self.Lmax:
            raise ValueError('ell index out of range')
        N = self.Nsizes[ell]
        if k >= N:
            raise ValueError('k index out of range')

        if not self._constructed:
            self._construct_basis()
        return self.etabasis[:,ell][:,np.newaxis] * self.sbasis[ell][k,:]


def plotfield(s, eta, f, fig=None, ax=None, stretch=False, aspect='equal', colorbar=True):
    """Plot a 2D slice of the field at phi = 0"""
    s, eta = s.ravel(), eta.ravel()
    ss, ee = s[np.newaxis,:], eta[:,np.newaxis]
    y = ee if stretch else np.sqrt(1-ss**2)*ee
    
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


def resize(mat, Lin, Nin, Lout, Nout, truncate=default_truncate):
    """Reshape the matrix from codomain size (Lin,Nin) to size (Lout,Nout).
       This appends and deletes rows as necessary without touching the columns.
       Nin and Nout are functions of ell and return the number of radial coefficients
       for each vertical degree"""
    if np.isscalar(Nin):
        Nin = Nsizes(Lin, Nin, truncate=truncate, functor=True)
    nintotal = sum([Nin(ell) for ell in range(Lin)])

    if np.isscalar(Nout):
        Nout = Nsizes(Lout, Nout, truncate=truncate, functor=True)
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


def Nsizes(Lmax, Nmax, truncate=default_truncate, functor=False):
    """Returns the number of radials coefficients for each vertical degree"""
    if truncate:
        _check_radial_degree(Lmax, Nmax)
    sizes = [Nmax - (ell//2 if truncate else 0) for ell in range(Lmax)]
    if functor:
        return lambda ell: sizes[ell]
    else:
        return sizes


def num_coeffs(Lmax, Nmax, truncate=default_truncate):
    """Return the total number of coefficients for a field"""
    return sum(Nsizes(Lmax, Nmax, truncate=truncate))


def _check_radial_degree(Lmax, Nmax):
    if Nmax < Lmax//2:
        raise ValueError('Radial degree too small for triangular truncation')


def _hstack(*args, **kwargs):
    return sparse.hstack(*args, **kwargs, format='csr')


def make_operator(dell, zop, sop, m, Lmax, Nmax, alpha, sigma, Lpad=0, Npad=0, truncate=default_truncate):
    """Kronecker the operator, always triangular truncating"""
    Nin_sizes = Nsizes(Lmax, Nmax, truncate=truncate)
    Nout_sizes = Nsizes(Lmax+Lpad, Nmax+Npad, truncate=truncate)
    Nin_offsets = np.append(0, np.cumsum(Nin_sizes))
    Nout_offsets = np.append(0, np.cumsum(Nout_sizes))

    oprows, opcols, opdata = [], [], []
    if dell < 0:
        ellmin = -dell
        ellmax = Lmax + min(Lpad, -dell)
    else:
        ellmin = 0
        ellmax = Lmax - dell
    for i in range(ellmin, ellmax):
        ellin, ellout = i+dell, i
        Nin, Nout = Nin_sizes[ellin], Nout_sizes[ellout]
        smat = sop(Nin, ellin+alpha+1/2, m+sigma)[:Nout,:]
        mat = sparse.csr_matrix(zop[ellin-max(dell,0)] * smat)

        matrows, matcols = mat.nonzero()
        oprows += (Nout_offsets[ellout] + matrows).tolist()
        opcols += (Nin_offsets[ellin] + matcols).tolist()
        opdata += np.asarray(mat[matrows,matcols]).ravel().tolist()

    return sparse.csr_matrix((opdata, (oprows, opcols)), shape=(Nout_offsets[-1],Nin_offsets[-1]))


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
    def __init__(self, codomain, dtype, internal, truncate):
        self._codomain = codomain

        if (np.zeros(1,dtype=dtype) + np.zeros(1,dtype=internal)).dtype == np.zeros(1,dtype=dtype).dtype:
            # Promote internal to the widest type
            internal = dtype
        self.dtype = dtype
        self.internal = internal
        self.truncate = truncate

        self.A = Jacobi.operator('A', dtype=internal)
        self.B = Jacobi.operator('B', dtype=internal)
        self.C = Jacobi.operator('C', dtype=internal)
        self.D = Jacobi.operator('D', dtype=internal)
        self.Z = Jacobi.operator('Z', dtype=internal)
        self.Id = Jacobi.operator('Id', dtype=internal)

        if self.truncate:
            self.n_for_ell = lambda Nmax, ell: Nmax - ell//2
        else:
            self.n_for_ell = lambda Nmax, ell: Nmax

    @property
    def codomain(self):
        return self._codomain


class Boundary(Operator):
    """Evaluate a field on the ball boundary"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate):
        Operator.__init__(self, codomain=None, dtype=dtype, internal=internal, truncate=truncate)

    def codomain(self, m, Lmax, Nmax, alpha, sigma):
        L = Lmax-1
        return (Nmax+L//2, L//2+alpha+1/2, m+sigma), (Nmax+(L-1)//2, (L+1)//2+alpha+1/2, m+sigma)

    def __call__(self, m, Lmax, Nmax, alpha, sigma, separate=False):
        if self.truncate:
            _check_radial_degree(Lmax, Nmax)
        A = self.A
        L = Lmax-1

        bc = Jacobi.polynomials(Lmax,alpha,alpha,1.,dtype=self.internal)

        nrows = [Nmax, Nmax] if self.truncate else [Nmax+L//2, Nmax+(L-1)//2]
        ncols = np.sum([self.n_for_ell(Nmax, ell) for ell in range(Lmax)])
        matrix = lambda shape: sparse.lil_matrix(shape, dtype=self.internal)
        Opeven, Opodd = tuple(matrix((nr,ncols)) for nr in nrows)
        index = 0
        for ell in range(Lmax):
            N = self.n_for_ell(Nmax, ell)
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
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate):
        Operator.__init__(self, codomain=Codomain(0,+1,+1), dtype=dtype, internal=internal, truncate=truncate)

    def __call__(self, m, Lmax, Nmax, alpha, sigma):
        def make_op(dell, zop, sop, Lpad=0, Npad=0):
            return make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

        A, B = self.A, self.B
        opz = (A(+1) @ B(+1))(Lmax,alpha,alpha)  # (ell,alpha,alpha) -> (ell,alpha+1,alpha+1)
        gamma_ell = opz.diagonal(0)
        delta_ell = opz.diagonal(2)
 
        Npad = 0 if self.truncate else 1
        Op1 = make_op(dell=0, zop=gamma_ell, sop=A(+1), Npad=Npad)  # (n,a,b) -> (n,a+1,b)
        Op2 = make_op(dell=2, zop=delta_ell, sop=A(-1), Npad=Npad)  # (n,a,b) -> (n+1,a-1,b)
 
        Op = Op1 + Op2
        return Op.astype(self.dtype)


class RadialVector(Operator):
    """Extract the spherical radial part of a velocity field"""   
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate):
        codomain = [Codomain(0,+1,0), Codomain(0,0,0), Codomain(+1,+1,0)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate)

    def __call__(self, m, Lmax, Nmax, alpha, exact=False):
        def make_op(dell, zop, sop, sigma, Lpad=0, Npad=0):
            return make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

        A, B, Z, Id = self.A, self.B, self.Z, self.Id
        Lpad, Npad = (1,1) if (not self.truncate or exact) else (0,0)

        # Coeff space operator: s * u(+)
        zop = np.ones(Lmax)
        sop = B(-1)                                      # (n,a,b) -> (n+1,a,b-1)
        Opp = 1/2 * make_op(dell=0, zop=zop, sop=sop, sigma=+1, Npad=Npad, Lpad=Lpad)
 
        # Coeff space operator: s * u(-)
        zop = np.ones(Lmax)
        sop = B(+1)                                      # (n,a,b) -> (n,a,b+1)
        Opm = 1/2 * make_op(dell=0, zop=zop, sop=sop, sigma=-1, Npad=Npad, Lpad=Lpad)
 
        # Coeff space operator: z * w = eta * (1-s**2)**0.5 * w
        opz = Z(Lmax,alpha,alpha)                        # (ell,alpha,alpha) -> (ell+1,alpha,alpha)
 
        zop = opz.diagonal(-1)
        sop = A(+1)                                      # (n,a,b) -> (n,a+1,b)
        Opz1 = make_op(dell=-1, zop=zop, sop=sop, sigma=0, Lpad=Lpad, Npad=Npad)
 
        zop = opz.diagonal(+1)
        sop = A(-1)                                      # (n,a,b) -> (n+1,a-1,b)
        Opz2 = make_op(dell=+1, zop=zop, sop=sop, sigma=0, Lpad=Lpad, Npad=Npad)

        Opz = 1/np.sqrt(2) * (Opz1 + Opz2)
        Op = _hstack([Opp, Opm, Opz])
        return Op.astype(self.dtype)


class RadialMultiplication(Operator):
    """Multiply a scalar field by the spherical radius vector"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate):
        codomain = [Codomain(0,0,0), Codomain(0,+1,0), Codomain(+1,+1,0)] 
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate)

    def __call__(self, m, Lmax, Nmax, alpha, exact=False):
        # Fixme: implement exact flag
        sigma = 0
        def make_op(dell, zop, sop, Lpad=0, Npad=0):
            return make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

        A, B, Z, Id = self.A, self.B, self.Z, self.Id
        Lpad, Npad = (1,1) if (not self.truncate or exact) else (0,0)

        # u(+) operator
        zop = np.ones(Lmax)
        sop = B(+1)                          # (n,a,b) -> (n,a,b+1)
        Opp = 1/2 * make_op(dell=0, zop=zop, sop=sop)

        # u(-) operator
        zop = np.ones(Lmax)
        sop = B(-1)                          # (n,a,b) -> (n+1,a,b-1)
        Opm = 1/2 * make_op(dell=0, zop=zop, sop=sop, Npad=Npad)

        # u(z) operator
        opz = Z(Lmax,alpha,alpha)            # (ell,alpha,alpha) -> (ell+1,alpha,alpha)
        zop = opz.diagonal(-1)
        sop = A(+1)                          # (n,a,b) -> (n,a+1,b)
        Opz1 = make_op(dell=-1, zop=zop, sop=sop, Lpad=Lpad, Npad=Npad)

        zop = opz.diagonal(+1)
        sop = A(-1)                          # (n,a,b) -> (n+1,a-1,b)
        Opz2 = make_op(dell=+1, zop=zop, sop=sop, Lpad=Lpad, Npad=Npad)

        Opz = 1/np.sqrt(2) * (Opz1 + Opz2)

        return Opp.astype(self.dtype), Opm.astype(self.dtype), Opz.astype(self.dtype)


class OneMinusRadiusSquared(Operator):
    """Multiply a field by (1-r**2)"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate):
        codomain = Codomain(+2,+1,-1)
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate)

    def __call__(self, m, Lmax, Nmax, alpha, sigma, exact=False):
        def make_op(dell, zop, sop, Lpad=0, Npad=0):
            return make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

        A, B = self.A, self.B
        Lpad, Npad = (2,1) if (not self.truncate or exact) else (0,0)

        opz = (A(-1) @ B(-1))(Lmax,alpha,alpha)
        zop = opz.diagonal(0)
        sop = A(-1)          # (n,a,b) -> (n+1,a-1,b)
        Op1 = make_op(dell=0, zop=zop, sop=sop, Lpad=Lpad, Npad=Npad)

        zop = opz.diagonal(-2)
        sop = A(+1)          # (n,a,b) -> (n,a+1,b)
        Op2 = make_op(dell=-2, zop=zop, sop=sop, Lpad=Lpad, Npad=Npad)

        Op = 1/2*(Op1 + Op2)
        return Op.astype(self.dtype)


class Gradient(Operator):
    """Compute the gradient of a scalar field"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate):
        if truncate:
            codomain = [Codomain(0,0,+1), Codomain(0,0,+1), Codomain(-1,0,+1)]
        else:
            codomain = [Codomain(0,0,+1), Codomain(0,+1,+1), Codomain(-1,0,+1)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate)

    def __call__(self, m, Lmax, Nmax, alpha):
        sigma = 0
        def make_op(dell, zop, sop, Lpad=0, Npad=0):
            return make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

        A, B, C, D, Id = self.A, self.B, self.C, self.D, self.Id

        op = (A(+1) @ B(+1))(Lmax,alpha,alpha)
        gamma_ell =  op.diagonal(0)
        delta_ell = -op.diagonal(2)
    
        # e(+)^* . Grad
        Op1 = make_op(dell=0, zop=2*gamma_ell, sop=D(+1))  # (n,a,b) -> (n-1,a+1,b+1)
        Op2 = make_op(dell=2, zop=2*delta_ell, sop=C(-1))  # (n,a,b) -> (n,a-1,b+1)
        Opp = Op1 + Op2
    
        # e(-)^* . Grad
        Npad = 0 if self.truncate else 1
        Op1 = make_op(dell=0, zop=2*gamma_ell, sop=C(+1), Npad=Npad)  # (n,a,b) -> (n,a+1,b-1)
        Op2 = make_op(dell=2, zop=2*delta_ell, sop=D(-1), Npad=Npad)  # (n,a,b) -> (n+1,a-1,b-1)
        Opm = Op1 + Op2

        # e(z)^* . Grad
        Lpad = 0 if self.truncate else -1
        zmat = np.sqrt(2) * D(+1)(Lmax,alpha,alpha)  # (ell,alpha,alpha) -> (ell-1,alpha+1,alpha+1)
        Opz = make_op(dell=1, zop=zmat.diagonal(1), sop=Id, Lpad=Lpad)

        return Opp.astype(self.dtype), Opm.astype(self.dtype), Opz.astype(self.dtype)


class Divergence(Operator):
    """Compute the divergence of a vector field"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate):
        if truncate:
            codomain = [Codomain(0,0,+1), Codomain(0,0,+1), Codomain(-1,0,+1)]
        else:
            codomain = [Codomain(0,+1,+1), Codomain(0,0,+1), Codomain(-1,0,+1)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate)
     
    def __call__(self, m, Lmax, Nmax, alpha):
        def make_op(dell, zop, sop, sigma, Lpad=0, Npad=0):
            return make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

        A, B, C, D, Id = self.A, self.B, self.C, self.D, self.Id
        Npad = 0 if self.truncate else 1

        op = (A(+1) @ B(+1))(Lmax,alpha,alpha)  # (ell,alpha,alpha) -> (ell,alpha+1,alpha+1)
        gamma_ell =  op.diagonal(0)
        delta_ell = -op.diagonal(2)
   
        # Div . e(+)^* .
        Op1 = make_op(dell=0, zop=2*gamma_ell, sop=C(+1), sigma=+1, Npad=Npad)  # (n,a,b) -> (n,a+1,b-1)
        Op2 = make_op(dell=2, zop=2*delta_ell, sop=D(-1), sigma=+1, Npad=Npad)  # (n,a,b) -> (n+1,a-1,b-1)
        Opp = Op1 + Op2

        # Div . e(-)^* . 
        Op1 = make_op(dell=0, zop=2*gamma_ell, sop=D(+1), sigma=-1, Npad=Npad)  # (n,a,b) -> (n-1,a+1,b+1)
        Op2 = make_op(dell=2, zop=2*delta_ell, sop=C(-1), sigma=-1, Npad=Npad)  # (n,a,b) -> (n,a-1,b+1)
        Opm = Op1 + Op2
 
        # Div . e(z)^* .
        zmat = np.sqrt(2) * D(+1)(Lmax,alpha,alpha)       # (ell,alpha,alpha) -> (ell-1,alpha+1,alpha+1)
        Opz = make_op(dell=1, zop=zmat.diagonal(1), sop=Id, sigma=0, Npad=Npad)
 
        Op = _hstack([Opp, Opm, Opz])
        return Op.astype(self.dtype)


class Curl(Operator):
    """Compute the divergence of a vector field"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate):
        if truncate:
            codomain = [Codomain(0,0,+1), Codomain(0,0,+1), Codomain(0,0,+1)]
        else:
            codomain = [Codomain(0,0,+1), Codomain(0,+1,+1), Codomain(0,+1,+1)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate)
     
    def __call__(self, m, Lmax, Nmax, alpha):
        def make_op(dell, zop, sop, sigma, Lpad=0, Npad=0):
            return make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

        A, B, C, D, Id = self.A, self.B, self.C, self.D, self.Id
        Lpad, Npad = (0,0) if self.truncate else (1,1)

        op = (A(+1) @ B(+1))(Lmax,alpha,alpha)  # (ell,alpha,alpha) -> (ell,alpha+1,alpha+1)
        gamma_ell =  op.diagonal(0)
        delta_ell = -op.diagonal(2)

        # e(+)^* . Curl
        zmat = np.sqrt(2) * D(+1)(Lmax,alpha,alpha)   # (ell,alpha,alpha) -> (ell-1,alpha+1,alpha+1)
        Opp_p = make_op(dell=1, zop=zmat.diagonal(1), sop=Id, sigma=+1)

        Op1 = make_op(dell=0, zop=-2*gamma_ell, sop=D(+1), sigma=0)  # (n,a,b) -> (n-1,a+1,b+1)
        Op2 = make_op(dell=2, zop=-2*delta_ell, sop=C(-1), sigma=0)  # (n,a,b) -> (n,a-1,b+1)
        Opp_z = Op1 + Op2

        # e(-)^* . Curl
        zmat = -np.sqrt(2) * D(+1)(Lmax,alpha,alpha)  # (ell,alpha,alpha) -> (ell-1,alpha+1,alpha+1)
        Opm_m = make_op(dell=1, zop=zmat.diagonal(1), sop=Id, sigma=-1, Npad=Npad)

        Op1 = make_op(dell=0, zop=2*gamma_ell, sop=C(+1), sigma=0, Npad=Npad)  # (n,a,b) -> (n,a+1,b-1)
        Op2 = make_op(dell=2, zop=2*delta_ell, sop=D(-1), sigma=0, Npad=Npad)  # (n,a,b) -> (n+1,a-1,b-1)
        Opm_z = Op1 + Op2

        # e(z)^* . Curl
        Op1 = make_op(dell=0, zop=-2*gamma_ell, sop=C(+1), sigma=+1, Npad=Npad)  # (n,a,b) -> (n,a+1,b-1)
        Op2 = make_op(dell=2, zop=-2*delta_ell, sop=D(-1), sigma=+1, Npad=Npad)  # (n,a,b) -> (n+1,a-1,b-1)
        Opz_p = Op1 + Op2

        Op1 = make_op(dell=0, zop=2*gamma_ell, sop=D(+1), sigma=-1, Npad=Npad)  # (n,a,b) -> (n-1,a+1,b+1)
        Op2 = make_op(dell=2, zop=2*delta_ell, sop=C(-1), sigma=-1, Npad=Npad)  # (n,a,b) -> (n,a-1,b+1)
        Opz_m = Op1 + Op2

        Zp, Zm, Zz = 0*Opp_p, 0*Opm_m, 0*Opz_p
        return 1j*(_hstack([Opp_p, Zp, Opp_z]).astype(self.dtype)), \
               1j*(_hstack([Zm, Opm_m, Opm_z]).astype(self.dtype)), \
               1j*(_hstack([Opz_p, Opz_m, Zz]).astype(self.dtype))


class ScalarLaplacian(Operator):
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate):
        Operator.__init__(self, codomain=Codomain(0,+1,+2), dtype=dtype, internal=internal, truncate=truncate)

    def __call__(self, m, Lmax, Nmax, alpha):
        kwargs = {'dtype':self.internal, 'internal':self.internal, 'truncate':self.truncate}
        divergence, gradient = Divergence(**kwargs), Gradient(**kwargs)

        if self.truncate:
            gradp, gradm, gradz = gradient(m, Lmax, Nmax, alpha)
            grad = sparse.vstack([gradp,gradm,gradz])
            div = divergence(m, Lmax, Nmax, alpha+1)
            dg = div @ grad
        else:
            gradp, gradm, gradz = gradient(m, Lmax, Nmax, alpha)
            gradp = resize(gradp, Lmax, Nmax, Lmax, Nmax+1, truncate=False)
            gradz = resize(gradz, Lmax-1, Nmax, Lmax, Nmax+1, truncate=False)
            grad = sparse.vstack([gradp,gradm,gradz])
            div = divergence(m, Lmax, Nmax+1, alpha+1)
            dg = div @ grad
            dg = resize(dg, Lmax, Nmax+2, Lmax, Nmax+1, truncate=False)
        return dg.astype(self.dtype)


class VectorLaplacian(Operator):
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate):
        if truncate:
            codomain = [Codomain(0,0,+2), Codomain(0,0,+2), Codomain(0,0,+2)]
        else:
            codomain = [Codomain(0,+1,+2), Codomain(0,+1,+2), Codomain(0,+1,+2)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate)

    def __call__(self, m, Lmax, Nmax, alpha):
        kwargs = {'dtype':self.internal, 'internal':self.internal, 'truncate':self.truncate}
        divergence, gradient, curl = Divergence(**kwargs), Gradient(**kwargs), Curl(**kwargs)

        if self.truncate:
            # Curl(Curl)
            curlp1, curlm1, curlz1 = curl(m, Lmax, Nmax, alpha)
            curlp2, curlm2, curlz2 = curl(m, Lmax, Nmax, alpha+1)

            # Grad(Div)
            div = divergence(m, Lmax, Nmax, alpha)
            gradp, gradm, gradz = gradient(m, Lmax, Nmax, alpha+1)

        else:
            # Curl(Curl)
            curlp1, curlm1, curlz1 = curl(m, Lmax, Nmax, alpha)
            curlp1 = resize(curlp1, Lmax, Nmax, Lmax, Nmax+1, truncate=False)
            curlp2, curlm2, curlz2 = curl(m, Lmax, Nmax+1, alpha+1)
            curlp2 = resize(curlp2, Lmax, Nmax+1, Lmax, Nmax+2, truncate=False)

            # Grad(Div)
            div = divergence(m, Lmax, Nmax, alpha)
            gradp, gradm, gradz = gradient(m, Lmax, Nmax+1, alpha+1)
            gradp = resize(gradp, Lmax, Nmax+1, Lmax, Nmax+2, truncate=False)
            gradz = resize(gradz, Lmax-1, Nmax+1, Lmax, Nmax+2, truncate=False)

        # Stack the operators
        curl1 = sparse.vstack([curlp1,curlm1,curlz1])
        curl2 = sparse.vstack([curlp2,curlm2,curlz2])
        grad = sparse.vstack([gradp,gradm,gradz])

        # Vector Laplacian
        cc = (curl2 @ curl1).real
        gd = grad @ div
        op = gd - cc

        # Prune entries near zero
        rows, cols, _ = sparse.find(abs(op) >= 1e-12)
        values = [op[r,c] for r,c in zip(rows, cols)]
        op = sparse.csr_matrix((values,(rows,cols)), shape=np.shape(op)).astype(self.dtype)

        # Extract sub-operators
        nin = num_coeffs(Lmax, Nmax, truncate=self.truncate)
        nout = num_coeffs(Lmax, Nmax+(0 if self.truncate else 2), truncate=self.truncate)
        Opp, Opm, Opz = op[:nout,:nin], op[nout:2*nout,nin:2*nin], op[2*nout:,2*nin:]

        if not self.truncate:
            Opp = resize(Opp, Lmax, Nmax+2, Lmax, Nmax+1, truncate=False)
            Opm = resize(Opm, Lmax, Nmax+2, Lmax, Nmax+1, truncate=False)
            Opz = resize(Opz, Lmax, Nmax+2, Lmax, Nmax+1, truncate=False)

        return Opp, Opm, Opz


def operator(name, field=None, dtype='float64', internal=internal_dtype, truncate=default_truncate):
    dispatch = lambda klass: klass(dtype=dtype, internal=internal, truncate=truncate)

    if name in ['divergence', 'div']:
        return dispatch(Divergence)
    if name in ['gradient', 'grad']:
        return dispatch(Gradient)
    if name == 'curl':
        return dispatch(Curl)
    if name in ['laplacian', 'lap']:
        if field in ['vector', 'vec']:
            op = dispatch(VectorLaplacian)
        else:
            op = dispatch(ScalarLaplacian)
        return op
    if name == 'rtimes':
        return dispatch(RadialMultiplication)
    if name == '1-r**2':
        return dispatch(OneMinusRadiusSquared)
    if name == 'rdot':
        return dispatch(RadialVector)
    if name in ['boundary', 'r=1']:
        return dispatch(Boundary)
    if name == 'conversion':
        return dispatch(Conversion)

    raise ValueError('Unknown operator')


def convert_alpha(ntimes, m, Lmax, Nmax, alpha, sigma, dtype='float64', internal=internal_dtype, truncate=default_truncate, exact=True):
    Conv = operator('conversion', dtype=internal, internal=internal, truncate=truncate)

    ncoeffs = sum(Nsizes(Lmax, Nmax, truncate=truncate))
    op = sparse.eye(ncoeffs, format='csr', dtype=internal)
    for i in range(ntimes):
        op1 = Conv(m, Lmax, Nmax+(0 if truncate else i), alpha=alpha+i, sigma=sigma)
        op = op1 @ op
    if not truncate:
        Ntrunc = Nmax+1 if exact else Nmax
        op = resize(op, Lmax, Nmax+ntimes, Lmax, Ntrunc, truncate=False)
    return op.astype(dtype)
    

def convert_beta(m, Lmax, Nmax, alpha, sigma, beta, dtype='float64', internal=internal_dtype, truncate=default_truncate):
    zop = np.ones(Lmax)
    A = Jacobi.operator('A', dtype=internal)
    sop = lambda n,a,b: (A(+1)**beta)(n,a-beta,b)
    op = make_operator(dell=0, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, truncate=truncate)
    return op.astype(dtype)

