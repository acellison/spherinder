import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import multiprocessing as mp

from dedalus_sphere import jacobi as Jacobi
from dedalus_sphere import operators
from .config import internal_dtype, max_processes, default_truncate, default_normalize


class Basis():
    """
    Spherinder basis element for fixed m.  Constructs the hierarchy of polynomial bases up to
    maximum vertical degree Lmax and untruncated radial degree Nmax.

    Parameters
    ----------
    s : np.ndarray
        Radial s coordinate values at which to evaluate the basis functions
    eta : np.ndarray
        Vertical eta coordinate values at which to evaluate the basis functions
    m : integer
        Azimuthal mode number m
    Lmax, Nmax : integer
        Maximum vertical and radial polynomial degrees, respectively
    sigma : integer, {-1,0,+1}
        Spin weight for the basis functions
    alpha : float
        Hierarchy parameter for basis functions.  Must be larger than -1
    galerkin : bool, optional
        Flag to use the Galerkin basis in which all functions are multiplied by 1-r**2
        so that they vanish on the spherical boundary
    truncate : bool, optional
        Flag to indicate using the triangular truncated expansion
    normalize : bool, optional
        Flag to indicate whether to normalize the basis functions with the s-to-t conversion
    dtype : np.dtype or str, optional
        Data type for the output of basis functions
    internal : np.dtype or str, optional
        Internal data type for computation
    lazy : bool, optional
        Flag to construct the basis functions only as they are requested
    parallel : bool, optional
        Flag to construct the basis functions in parallel

    """
    def __init__(self, s, eta, m, Lmax, Nmax, sigma=0, alpha=0, galerkin=False, truncate=default_truncate, normalize=default_normalize,
                 dtype='float64', internal='float128', lazy=True, parallel=False):
        self.s, self.eta = s, eta
        self.m, self.Lmax, self.Nmax = m, Lmax, Nmax
        self.sigma, self.alpha = sigma, alpha
        self.dtype, self.internal = dtype, internal
        self.galerkin = galerkin
        self.parallel = parallel
        self.truncate = truncate
        self.normalize = normalize
        self.Nsizes = Nsizes(self.Lmax, self.Nmax, truncate=self.truncate)

        if not lazy:
            self._construct_basis()
        else:
            self._sbasis, self._etabasis = None, None
            self._constructed = False

    @property
    def ncoeffs(self):
        """Total number of coefficients for the given Lmax and Nmax with specified truncation"""
        return num_coeffs(self.Lmax, self.Nmax, truncate=self.truncate)


    @property
    def norm_scale(self):
        """Scale factor in t direction for normalization"""
        return np.sqrt(2**(2+self.alpha+1/2)) if self.normalize else 1.


    class _Constructor():
        """Helper object to parallelize basis construction"""
        def __init__(self, Nsizes, m, sigma, alpha, t, onept, onemt, truncate, normscale, internal, dtype):
            self.Nsizes, self.m, self.sigma, self.alpha = Nsizes, m, sigma, alpha
            self.truncate, self.normscale, self.internal, self.dtype = truncate, normscale, internal, dtype
            self.t, self.onept, self.onemt = t, onept, onemt
        def __call__(self, ell):
            m, sigma, alpha = self.m, self.sigma, self.alpha
            internal, dtype = self.internal, self.dtype
            t, onept, onemt = self.t, self.onept, self.onemt
            N = self.Nsizes[ell]
            scale = self.normscale
            return scale * ((onept * onemt**ell) * Jacobi.polynomials(N,ell+alpha+1/2,m+sigma,t,dtype=internal)).astype(dtype)


    def _construct_basis(self):
        m, Lmax, Nmax = self.m, self.Lmax, self.Nmax
        sigma, alpha = self.sigma, self.alpha
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
        fun = Basis._Constructor(self.Nsizes, m, sigma, alpha, t, onept, onemt, truncate, self.norm_scale, internal, dtype)
        if self.parallel:
            num_processes = min(mp.cpu_count(), max_processes)
            pool = mp.Pool(num_processes)
            sbasis = pool.map(fun, range(Lmax))
        else:
            sbasis = [fun(ell) for ell in range(Lmax)]

        self._sbasis, self._etabasis = sbasis, etabasis
        self._constructed = True


    def expand(self, coeffs):
        """
        Expand a field as a sum of basis functions times coefficients.

        Parameters
        ----------
        self : Basis
            Basis function object
        coeffs : np.ndarray
            Array of coefficients of size specified by the Basis construction
        """
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

    @property
    def sbasis(self):
        """List of array of radial basis functions, one set for each ell"""
        if not self._constructed:
            self._construct_basis()
        return self._sbasis

    @property
    def etabasis(self):
        """Array of vertical basis function, one for each ell"""
        if not self._constructed:
            self._construct_basis()
        return self._etabasis


    def __getitem__(self, index):
        """
        Access a basis function at the specified (ell, k) index

        Parameters
        ----------
        self : Basis
            Basis function object
        index : tuple of integers
            (ell, k) index for the desired basis function
        """
        ell, k = index[0], index[1]
        if ell >= self.Lmax:
            raise ValueError('ell index out of range')
        N = self.Nsizes[ell]
        if k >= N:
            raise ValueError('k index out of range')

        if not self._constructed:
            self._construct_basis()
        return self.etabasis[:,ell][:,np.newaxis] * self.sbasis[ell][k,:]



def Nsizes(Lmax, Nmax, truncate=default_truncate, functor=False):
    """
    Returns the number of radials coefficients for each vertical degree

    Parameters
    ----------
    Lmax : int
        Maximum vertical degree
    Nmax : int
        Maximum radial degree
    truncate : bool, optional
        Flag to specify triangular truncation
    functor : bool, optional
        If True, returns a function object that takes the vertical degree
        ell and returns the number of radial coefficients for that degree.
        If False, returns a list of radial sizes, one per vertical degree.

    Returns
    -------
    radial_sizes : list or function
        If functor is False, a list of radial coefficient sizes for each ell.
        If functor is True, a function object that returns the radial coefficient
        size for an input vertical degree ell.
    """
    if truncate:
        _check_radial_degree(Lmax, Nmax)
    sizes = [Nmax - (ell//2 if truncate else 0) for ell in range(Lmax)]
    return (lambda ell: sizes[ell]) if functor else sizes


def num_coeffs(Lmax, Nmax, truncate=default_truncate):
    """
    Return the total number of coefficients for a field

    Parameters
    ----------
    Lmax : int
        Maximum vertical degree
    Nmax : int
        Maximum radial degree
    truncate : bool, optional
        Flag to specify triangular truncation

    Returns
    -------
    num_coeffs : int
        Total number of coefficients for specified triangular truncation
    """
    return sum(Nsizes(Lmax, Nmax, truncate=truncate))


def coeff_sizes(Lmax, Nmax, truncate=default_truncate):
    """
    Return the number of radial coefficients for each vertical degree,
    and the offsets for indexing into a coefficient vector for the first
    radial mode of each vertical degree

    Parameters
    ----------
    Lmax : int
        Maximum vertical degree
    Nmax : int
        Maximum radial degree
    truncate : bool, optional
        Flag to specify triangular truncation

    Returns
    -------
    lengths : np.ndarray
        Array of size Lmax with the number of radial coefficients for each ell
    offsets : np.ndarray
        Array of offsets to the radial coefficients for each vertical degree ell
    """
    lengths = Nsizes(Lmax, Nmax, truncate=truncate)
    offsets = np.append(0, np.cumsum(lengths)[:-1])
    return lengths, offsets


def norm_ratio(dalpha, normalize=default_normalize):
    """
    Ratio of basis normalization scale factor for a change in alpha

    Parameters
    ----------
    dalpha : float
        Change in alpha from the input parameter to the output parameter
        for a given operator
    normalize : bool, optional
        Flag for s-to-t normalization

    Returns
    -------
    ratio : float
        Normalization ratio
    """
    return 2**(-dalpha/2) if normalize else 1


def _check_radial_degree(Lmax, Nmax):
    """Check the radial degree is large enough for triangular truncation"""
    if Nmax < Lmax//2:
        raise ValueError('Radial degree too small for triangular truncation')


def _hstack(*args, **kwargs):
    """Wrapper around sparse.hstack to return results in a csr_matrix"""
    return sparse.hstack(*args, **kwargs, format='csr')


def _make_operator(dell, zop, sop, m, Lmax, Nmax, alpha, sigma, Lpad=0, Npad=0, truncate=default_truncate):
    """
    Kronecker the operator in the eta and s directions

    Parameters
    ----------
    dell : integer
        Change in ell from input to output
    zop : np.ndarray
        Coefficients for the operator in the eta coordinate
    sop : dedalus_sphere.jacobi.JacobiOperator
        Operator object for generating the coefficients of the radial
        part of spherinder operator.  Takes parameters (n,a,b) corresponding
        to radial degree n and Jacobi polynomial parameters a,b
    m : integer
        Azimuthal wave number
    Lmax : integer
        Maximum vertical degree
    Nmax : integer
        Maximum radial degree
    alpha : float
        Basis function hierarchy parameter, must be larger than -1
    sigma : float, {-1,0,+1}
        Basis function spin weight
    Lpad : integer, optional
        Change in maximum vertical degree from input to output
    Npad : integer, optional
        Change in maximum radial degree from input to output
    truncate : bool, optional
        Flag to specify triangular truncation

    Returns
    -------
    operator : sparse.csr_matrix
        Sparse matrix representation of the operator
    """
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
    """
    Codomain object for the action of operators.  This object encodes
    the change in vertical degree (dell), radial degree (dn) and hierarchy
    parameter (dalpha) as a result of applying a given operator.

    Parameters
    ----------
    dell : int
        Change in maximum vertical degree
    dn : int
        Change in maximum radial degree
    dalpha : int
        Change in hierarchy parameter alpha
    """
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
    """
    Base class for representing operators on sets of spherinder basis function.
    Calling a derived Operator with parameters (m, Lmax, Nmax, alpha, [sigma])
    builds the operator in sparse matrix form as the left action on a set of 
    coefficients.

    Parameters
    ----------
    codomain : Codmain
        codomain or list of codomains for the derived operator
    dtype : string or np.dtype
        Data type for the output of operator construction
    internal : string or np.dtype
        Data type for the computation of operators
    normalize : bool
        Flag for triangular truncation
    """
    def __init__(self, codomain, dtype, internal, truncate, normalize):
        self._codomain = codomain

        if (np.zeros(1,dtype=dtype) + np.zeros(1,dtype=internal)).dtype == np.zeros(1,dtype=dtype).dtype:
            # Promote internal to the widest type
            internal = dtype
        self.dtype = dtype
        self.internal = internal
        self.truncate = truncate
        self.normalize = normalize

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
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize):
        Operator.__init__(self, codomain=None, dtype=dtype, internal=internal, truncate=truncate, normalize=normalize)

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
        ncols = num_coeffs(Lmax, Nmax, truncate=self.truncate)
        zeros = lambda shape: sparse.lil_matrix(shape, dtype=self.internal)
        Opeven, Opodd = tuple(zeros((nr,ncols)) for nr in nrows)
        index, nsizes = 0, Nsizes(Lmax, Nmax, truncate=self.truncate)
        for ell in range(Lmax):
            N = nsizes[ell]
            op = bc[ell] * ((A(+1)**((L-ell)//2)) @ (A(-1)**(ell//2)))(N, ell+alpha+1/2, m+sigma)
            mat = [Opeven, Opodd][ell % 2]
            mat[:np.shape(op)[0],index:index+N] = op
            index += N

        norm_scale = np.sqrt(2**(2+alpha+1/2)) if self.normalize else 1.
        Opeven, Opodd = [(norm_scale * mat).astype(self.dtype) for mat in [Opeven, Opodd]]
        if separate:
            return Opeven, Opodd
        else:
            return sparse.vstack([Opeven,Opodd], format='csr')


class Conversion(Operator):
    """Convert up in alpha index.  This isn't really a tensor operation since it can
       act independently on components of vectors"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize):
        if truncate:
            codomain = Codomain(0,0,+1)
        else:
            codomain = Codomain(0,+1,+1)
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate, normalize=normalize)

    def __call__(self, m, Lmax, Nmax, alpha, sigma):
        def make_op(dell, zop, sop, Lpad=0, Npad=0):
            return _make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

        A, B = self.A, self.B
        opz = (A(+1) @ B(+1))(Lmax,alpha,alpha)  # (ell,alpha,alpha) -> (ell,alpha+1,alpha+1)
 
        Npad = 0 if self.truncate else 1
        Op1 = make_op(dell=0, zop=opz.diagonal(0), sop=A(+1), Npad=Npad)  # (n,a,b) -> (n,a+1,b)
        Op2 = make_op(dell=2, zop=opz.diagonal(2), sop=A(-1), Npad=Npad)  # (n,a,b) -> (n+1,a-1,b)
 
        Op = Op1 + Op2

        scale = norm_ratio(dalpha=+1, normalize=self.normalize)
        return (scale * Op).astype(self.dtype)


class RadialComponent(Operator):
    """Extract the spherical radial part of a velocity field"""   
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize):
        codomain = [Codomain(0,+1,0), Codomain(0,0,0), Codomain(+1,+1,0)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate, normalize=normalize)

    def __call__(self, m, Lmax, Nmax, alpha, exact=False):
        def make_op(dell, zop, sop, sigma, Lpad=0, Npad=0):
            return _make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

        A, B, Z, Id = self.A, self.B, self.Z, self.Id
        Lpad, Npad = (1,1) if (not self.truncate or exact) else (0,0)

        # Coeff space operator: s * u(+)
        Opp = 1/2 * make_op(dell=0, zop=np.ones(Lmax), sop=B(-1), sigma=+1, Npad=Npad, Lpad=Lpad)  # (n,a,b) -> (n+1,a,b-1)
 
        # Coeff space operator: s * u(-)
        Opm = 1/2 * make_op(dell=0, zop=np.ones(Lmax), sop=B(+1), sigma=-1, Npad=Npad, Lpad=Lpad)  # (n,a,b) -> (n,a,b+1)
 
        # Coeff space operator: z * w = eta * (1-s**2)**0.5 * w
        opz = Z(Lmax,alpha,alpha)                        # (ell,alpha,alpha) -> (ell+1,alpha,alpha)
        Opz1 = make_op(dell=-1, zop=opz.diagonal(-1), sop=A(+1), sigma=0, Lpad=Lpad, Npad=Npad)  # (n,a,b) -> (n,a+1,b)
        Opz2 = make_op(dell=+1, zop=opz.diagonal(+1), sop=A(-1), sigma=0, Lpad=Lpad, Npad=Npad)  # (n,a,b) -> (n+1,a-1,b)

        Opz = 1/np.sqrt(2) * (Opz1 + Opz2)
        Op = _hstack([Opp, Opm, Opz])
        return Op.astype(self.dtype)


class RadialMultiplication(Operator):
    """Multiply a scalar field by the spherical radius vector"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize):
        codomain = [Codomain(0,0,0), Codomain(0,+1,0), Codomain(+1,+1,0)] 
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate, normalize=normalize)

    def __call__(self, m, Lmax, Nmax, alpha, exact=False):
        sigma = 0
        def make_op(dell, zop, sop, Lpad=0, Npad=0):
            return _make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

        A, B, Z, Id = self.A, self.B, self.Z, self.Id
        Lpad, Npad = (1,1) if (not self.truncate or exact) else (0,0)

        # u(+) operator
        Opp = 1/2 * make_op(dell=0, zop=np.ones(Lmax), sop=B(+1))  # (n,a,b) -> (n,a,b+1)

        # u(-) operator
        Opm = 1/2 * make_op(dell=0, zop=np.ones(Lmax), sop=B(-1), Npad=Npad)  # (n,a,b) -> (n+1,a,b-1)

        # u(z) operator
        opz = Z(Lmax,alpha,alpha)            # (ell,alpha,alpha) -> (ell+1,alpha,alpha)
        Opz1 = make_op(dell=-1, zop=opz.diagonal(-1), sop=A(+1), Lpad=Lpad, Npad=Npad)  # (n,a,b) -> (n,a+1,b)
        Opz2 = make_op(dell=+1, zop=opz.diagonal(+1), sop=A(-1), Lpad=Lpad, Npad=Npad)  # (n,a,b) -> (n+1,a-1,b)

        Opz = 1/np.sqrt(2) * (Opz1 + Opz2)

        return Opp.astype(self.dtype), Opm.astype(self.dtype), Opz.astype(self.dtype)


class OneMinusRadiusSquared(Operator):
    """Multiply a field by (1-r**2)"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize):
        codomain = Codomain(+2,+1,-1)
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate, normalize=normalize)

    def __call__(self, m, Lmax, Nmax, alpha, sigma, exact=False):
        def make_op(dell, zop, sop, Lpad=0, Npad=0):
            return _make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

        A, B = self.A, self.B
        Lpad, Npad = (2,1) if (not self.truncate or exact) else (0,0)

        opz = (A(-1) @ B(-1))(Lmax,alpha,alpha)
        Op1 = make_op(dell=0,  zop=opz.diagonal(0),  sop=A(-1), Lpad=Lpad, Npad=Npad)    # (n,a,b) -> (n+1,a-1,b)
        Op2 = make_op(dell=-2, zop=opz.diagonal(-2), sop=A(+1), Lpad=Lpad, Npad=Npad)  # (n,a,b) -> (n,a+1,b)

        Op = 1/2*(Op1 + Op2)

        scale = norm_ratio(dalpha=-1, normalize=self.normalize)
        return (scale * Op).astype(self.dtype)


class Gradient(Operator):
    """Compute the gradient of a scalar field"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize):
        if truncate:
            codomain = [Codomain(0,0,+1), Codomain(0,0,+1), Codomain(-1,0,+1)]
        else:
            codomain = [Codomain(0,0,+1), Codomain(0,+1,+1), Codomain(-1,0,+1)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate, normalize=normalize)

    def __call__(self, m, Lmax, Nmax, alpha):
        sigma = 0
        def make_op(dell, zop, sop, Lpad=0, Npad=0):
            return _make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

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

        scale = norm_ratio(dalpha=+1, normalize=self.normalize)
        return (scale * Opp).astype(self.dtype), \
               (scale * Opm).astype(self.dtype), \
               (scale * Opz).astype(self.dtype)


class Divergence(Operator):
    """Compute the divergence of a vector field"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize):
        if truncate:
            codomain = [Codomain(0,0,+1), Codomain(0,0,+1), Codomain(-1,0,+1)]
        else:
            codomain = [Codomain(0,+1,+1), Codomain(0,0,+1), Codomain(-1,0,+1)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate, normalize=normalize)
     
    def __call__(self, m, Lmax, Nmax, alpha):
        def make_op(dell, zop, sop, sigma, Lpad=0, Npad=0):
            return _make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

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

        scale = norm_ratio(dalpha=+1, normalize=self.normalize)
        return (scale * Op).astype(self.dtype)


class Curl(Operator):
    """Compute the divergence of a vector field"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize):
        if truncate:
            codomain = [Codomain(0,0,+1), Codomain(0,0,+1), Codomain(0,0,+1)]
        else:
            codomain = [Codomain(0,0,+1), Codomain(0,+1,+1), Codomain(0,+1,+1)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate, normalize=normalize)
     
    def __call__(self, m, Lmax, Nmax, alpha):
        def make_op(dell, zop, sop, sigma, Lpad=0, Npad=0):
            return _make_operator(dell=dell, zop=zop, sop=sop, m=m, Lmax=Lmax, Nmax=Nmax, alpha=alpha, sigma=sigma, Lpad=Lpad, Npad=Npad, truncate=self.truncate)

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

        scale = norm_ratio(dalpha=+1, normalize=self.normalize)
        return 1j*((scale * _hstack([Opp_p, Zp, Opp_z])).astype(self.dtype)), \
               1j*((scale * _hstack([Zm, Opm_m, Opm_z])).astype(self.dtype)), \
               1j*((scale * _hstack([Opz_p, Opz_m, Zz])).astype(self.dtype))


class ScalarLaplacian(Operator):
    """Compute the laplacian of a scalar field"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize):
        if truncate:
            codomain = Codomain(0,0,+2)
        else:
            codomain = Codomain(0,+1,+2)
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate, normalize=normalize)

    def __call__(self, m, Lmax, Nmax, alpha):
        kwargs = {'dtype':self.internal, 'internal':self.internal, 'truncate':self.truncate, 'normalize':self.normalize}
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
    """Compute the laplacian of a vector field"""
    def __init__(self, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize):
        if truncate:
            codomain = [Codomain(0,0,+2), Codomain(0,0,+2), Codomain(0,0,+2)]
        else:
            codomain = [Codomain(0,+1,+2), Codomain(0,+1,+2), Codomain(0,+1,+2)]
        Operator.__init__(self, codomain=codomain, dtype=dtype, internal=internal, truncate=truncate, normalize=normalize)

    def __call__(self, m, Lmax, Nmax, alpha):
        kwargs = {'dtype':self.internal, 'internal':self.internal, 'truncate':self.truncate, 'normalize':self.normalize}
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


def operator(name, field=None, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize):
    """
    Construct an operator of the given name.  This is the standard method for
    constructing derived Operator objects.

    Parameters
    ----------
    name : str
        String name of the derived Operator
    field : str
        Ignored unless constructed the Laplacian operator.  In this case field
        should be one of ['scalar', 'vector'] to dispatch the appropriate operator
    dtype : np.dtype or str, optional
        Data type for the output of operator construction
    internal : np.dtype or str, optional
        Internal data type for computation
    truncate : bool, optional
        Flag to indicate using the triangular truncated expansion
    normalize : bool, optional
        Flag to indicate whether to normalize the basis functions with the s-to-t conversion

    Returns
    -------
    operator : Operator
        Derived Operator object that builds the matrix operator when called
    """
    dispatch = lambda klass: klass(dtype=dtype, internal=internal, truncate=truncate, normalize=normalize)

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
        return dispatch(RadialComponent)
    if name in ['boundary', 'r=1']:
        return dispatch(Boundary)
    if name == 'conversion':
        return dispatch(Conversion)

    raise ValueError('Unknown operator')


def convert_alpha(ntimes, m, Lmax, Nmax, alpha, sigma, dtype='float64', internal=internal_dtype, truncate=default_truncate, normalize=default_normalize, exact=True):
    """
    Construct the matrix operator converting basis functions form alpha to alpha+ntimes.

    Parameters
    ----------
    ntimes : integer
        Number of times to increment the alpha index
    m : integer
        Azimuthal mode number m
    Lmax, Nmax : integer
        Maximum vertical and radial polynomial degrees, respectively
    alpha : float
        Hierarchy parameter for basis functions.  Must be larger than -1
    sigma : integer, {-1,0,+1}
        Spin weight for the basis functions
    dtype : np.dtype or str, optional
        Data type for the output of operator construction
    internal : np.dtype or str, optional
        Internal data type for computation
    truncate : bool, optional
        Flag to indicate using the triangular truncated expansion
    normalize : bool, optional
        Flag to indicate whether to normalize the basis functions with the s-to-t conversion
    exact : bool, optional
        Flag for exact conversion.  Only used when truncate is False, otherwise alpha
        conversion is always exact

    Returns
    -------
    operator : sparse.csr_matrix
        Sparse matrix for alpha conversion
    """
    ntimes = int(ntimes)
    Conv = operator('conversion', dtype=internal, internal=internal, truncate=truncate, normalize=normalize)

    ncoeffs = sum(Nsizes(Lmax, Nmax, truncate=truncate))
    op = sparse.eye(ncoeffs, format='csr', dtype=internal)
    for i in range(ntimes):
        op1 = Conv(m, Lmax, Nmax+(0 if truncate else i), alpha=alpha+i, sigma=sigma)
        op = op1 @ op
    if not truncate:
        Ntrunc = Nmax+1 if exact else Nmax
        op = resize(op, Lmax, Nmax+ntimes, Lmax, Ntrunc, truncate=False)
    return op.astype(dtype)
    

def tau_projection(m, Lmax, Nmax, alpha, sigma, alpha_bc, shift=0, dtype='float64', internal=internal_dtype, truncate=default_truncate):
    """
    Create the tau projection matrix.  Converts the tau polynomial expressed in the alpha_bc basis
    to the alpha basis.

    Parameters
    ----------
    m : integer
        Azimuthal mode number m
    Lmax, Nmax : integer
        Maximum vertical and radial polynomial degrees, respectively
    alpha : float
        Hierarchy parameter for output space
    sigma : integer, {-1,0,+1}
        Spin weight for the basis functions
    alpha_bc : float
        Hierarchy parameter for input space
    shift : integer
        Number of additional highest modes for projection
    dtype : np.dtype or str, optional
        Data type for the output of operator construction
    internal : np.dtype or str, optional
        Internal data type for computation
    truncate : bool, optional
        Flag to indicate using the triangular truncated expansion

    Returns
    -------
    projection : scipy.sparse matrix
        Sparse matrix containing tua projection columns
    """
    # FIXME: shift should just return a single column, not all of them after shift!
    Conv = convert_alpha(alpha-alpha_bc, m, Lmax, Nmax, alpha=alpha_bc, sigma=sigma, dtype=dtype, internal=internal, truncate=truncate)
    lengths, offsets = coeff_sizes(Lmax, Nmax, truncate=truncate)
    indices = offsets+lengths-1
    col1 = sparse.hstack([Conv[:,indices[ell]-shift:indices[ell]+1] for ell in range(Lmax-2*(1+shift))])
    col2 = Conv[:,offsets[-2*(1+shift)]:]
    return sparse.hstack([col1, col2])


def resize(mat, Lin, Nin, Lout, Nout, truncate=default_truncate):
    """
    Reshape the matrix from codomain size (Lin,Nin) to size (Lout,Nout).
    This appends and deletes rows as necessary without touching the columns.
    Nin and Nout are functions of ell and return the number of radial coefficients
    for each vertical degree

    Parameters
    ----------
    mat : scipy.sparse matrix
        Sparse matrix with coefficient size (Lin,Nin)
    Lin, Nin : integer
        Maximum vertical and radial input polynomial degrees, respectively
    Lout, Nout : integer
        Maximum vertical and radial output polynomial degrees, respectively
    truncate : bool, optional
        Flag to indicate using the triangular truncated expansion

    Returns
    -------
    resized_matrix : scipy.sparse matrix
        Input matrix resized to coefficient size (Lout, Nout)
    """
    if np.isscalar(Nin):
        Nin = Nsizes(Lin, Nin, truncate=truncate, functor=True)
    nintotal = sum([Nin(ell) for ell in range(Lin)])

    if np.isscalar(Nout):
        Nout = Nsizes(Lout, Nout, truncate=truncate, functor=True)
    nouttotal = sum([Nout(ell) for ell in range(Lout)])

    # Check if all sizes match.  If so, just return the input matrix
    if Lin == Lout and all([Nin(ell) == Nout(ell) for ell in range(Lin)]):
        return mat

    # Check the number of rows matches the input (Lin, Nin) dimensions
    nrows, ncols = np.shape(mat)
    if not nintotal == nrows:
        raise ValueError('Incorrect size')

    # Extract the nonzero entries of the input matrix
    if not isinstance(mat, sparse.csr_matrix):
        mat = mat.tocsr()
    rows, cols = mat.nonzero()

    # If we have the zero matrix just return a zero matrix
    if len(rows) == 0:
        return sparse.lil_matrix((nouttotal, ncols))

    # Build up the resized operator
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
    """
    Chuck any identically-zero rows from the matrix

    Parameters
    ----------
    mat : scipy.sparse matrix
        Sparse matrix

    Returns
    -------
    nonzero_matrix : scipy.sparse.csr_matrix
        Copy of mat with any zero rows thrown out
    """
    rows, cols = mat.nonzero()
    zrows = list(set(range(np.shape(mat)[0])) - set(rows))
    if not zrows:
        return mat
    for z in zrows:
        i = np.argmax(rows > z)
        if i > 0:
            rows[i:] -= 1
    return sparse.csr_matrix((mat.data, (rows,cols)), shape=(max(rows)+1,np.shape(mat)[1]))


def eliminate_zeros(mat, tol=0.):
    """
    Prune zeros (or small values when tol is nonzero) from the sparse matrix

    Parameters
    ----------
    mat : scipy.sparse matrix
        Sparse matrix
    tol : float
        Value below which matrix entries are flushed to zero

    Returns
    -------
    nonzero_matrix : scipy.sparse.csr_matrix
        Copy of mat with small values set to zero
    """
    if tol <= 0:
        mat.eliminate_zeros()
        return mat
    mat = mat.tocsr()
    rows, cols, _ = sparse.find(abs(mat) >= tol)
    values = [mat[r,c] for r,c in zip(rows, cols)]
    return sparse.csr_matrix((values,(rows,cols)), shape=np.shape(mat))


def plotfield(s, eta, f, fig=None, ax=None, stretch=False, arcsin=False, aspect='equal', colorbar=True, cmap='RdBu', cbar_format=None, shading=None):
    """
    Plot a 2D slice of the field at phi = 0

    Parameters
    ----------
    s : np.ndarray
        Radial coordinate to plot.  Second dimension of f
    eta : np.ndarray
        Vertical coordinate to plot.  First dimenson of f
    f : np.ndarray
        2D array containing values for the field at eta,s
    fig : matplotlib.pyplot.Figure, optional
        Figure object to add plot
    ax : matplotlib.pyplot.Axes, optional
        Axes object to add plot
    stretch : bool, optional
        Flag to stretch the sphere into a cylinder shape,
        plotting in coordinates (s,eta) if True, otherwise (s,z).
    arcsin : bool, optional
        Flag to plot the field in the λ = arcsin(s) coordinate
    aspect : str, optional
        Aspect parameter for the Axes object
    colorbar : bool, optional
        If True, plot a colorbar alongside the axes
    cmap : str, optional
        Colormap identifier for plotting
    cbar_format : str, optional
        If 'log', print the colorbar labels in powers of 10
    shading : str, optional
        Shading style for matplotlib.pyplot.pcolormesh.
        Defaults to 'auto' for rectilinear coordinates, otherwise 'gouraud'.

    Returns
    -------
    fig, ax : matplotlib.pyplot Figure and Axes objects
        Figure and Axes with the plotted field
    """
    ss, ee = s.ravel()[np.newaxis,:], eta.ravel()[:,np.newaxis]
    y = ee if stretch else np.sqrt(1-ss**2)*ee
    if shading is None:
        shading = 'auto' if stretch else 'gouraud'

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(4.25,6))

    if arcsin:
        ss = np.arcsin(ss)
        slabel = r'$\arcsin{(s)}$'
    else:
        slabel = 's'

    im = ax.pcolormesh(ss, y, f, cmap=cmap, shading=shading)
    if colorbar:
        def fmt(x, pos):
            a, b = '{:.0e}'.format(x).split('e')
            b = int(b)
            return r'${} \times 10^{{{}}}$'.format(a, b)
        if cbar_format == 'log':
            cbar_format = ticker.FuncFormatter(fmt)
        fig.colorbar(im, ax=ax, format=cbar_format)

    ax.set_xlabel(slabel)
    ax.set_ylabel('η' if stretch else 'z')
    if aspect is not None:
        ax.set_aspect(aspect, adjustable='datalim')

    fig.set_tight_layout(True)
    return fig, ax

