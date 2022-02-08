from . import operators as sph
from .config import internal_dtype
from scipy import sparse


def horizontal_laplacian(m, Lmax, Nmax, alpha, dtype='float64', internal=internal_dtype):
    """Codomain: (Lmax->Lmax, Nmax->Nmax, alpha->alpha+2)"""
    kwargs = {'dtype': internal, 'internal': internal}
    divergence, gradient = sph.Divergence(**kwargs), sph.Gradient(**kwargs)
    num_coeffs = sph.num_coeffs(Lmax, Nmax)

    gradp, gradm, _ = gradient(m, Lmax, Nmax, alpha)
    div = divergence(m, Lmax, Nmax, alpha+1)

    grad = sparse.vstack([gradp,gradm])
    div = div[:,:2*num_coeffs]

    dg = div @ grad

    return dg.astype(dtype)


def horizontal_laplacian_squared(m, Lmax, Nmax, alpha, dtype='float64', internal=internal_dtype):
    lap1 = horizontal_laplacian(m, Lmax, Nmax, alpha,   dtype=internal_dtype, internal=internal_dtype)
    lap2 = horizontal_laplacian(m, Lmax, Nmax, alpha+2, dtype=internal_dtype, internal=internal_dtype)
    return (lap2 @ lap1).astype(dtype)


def S_squared(m, Lmax, Nmax, alpha, exact=False, dtype='float64', internal=internal_dtype):
    """Multiply a field by s**2 
       Codomain: (Lmax->Lmax, Nmax->Nmax+1, alpha->alpha) if exact
    """
    kwargs = {'dtype': internal, 'internal': internal}

    rmul = sph.RadialMultiplication(**kwargs)(m, Lmax, Nmax, alpha, exact=True)
    smulp, smulm = rmul[:2]
    smulp = sph.resize(smulp, Lmax, Nmax, Lmax, Nmax+1)

    rdot = sph.RadialComponent(**kwargs)(m, Lmax, Nmax+1, alpha, exact=True)
    sdot = rdot[:,:2*sph.num_coeffs(Lmax,Nmax+1)]

    op = sdot @ sparse.vstack([smulp, smulm])

    Nout = Nmax + (1 if exact else 0)
    op = sph.resize(op, Lmax+1, Nmax+2, Lmax, Nout)

    return op.astype(dtype)


def S_fourth(m, Lmax, Nmax, alpha, exact=False, dtype='float64', internal=internal_dtype):
    """Multiply a field by s**4
       Codomain: (Lmax->Lmax, Nmax->Nmax+2, alpha->alpha) if exact
    """
    s21 = S_squared(m, Lmax, Nmax,   alpha, exact=True, dtype=internal, internal=internal)
    s22 = S_squared(m, Lmax, Nmax+1, alpha, exact=True, dtype=internal, internal=internal)

    op = s22 @ s21
    if not exact:
        op = sph.resize(op, Lmax, Nmax+2, Lmax, Nmax)

    return op.astype(dtype)


def SdS(m, Lmax, Nmax, alpha, dtype='float64', internal=internal_dtype):
    """s d/ds operator
       Codomain: (Lmax->Lmax, Nmax->Nmax, alpha->alpha+1)
    """
    kwargs = {'dtype': internal, 'internal': internal}
    gradient = sph.Gradient(**kwargs)
    gradp, gradm = gradient(m, Lmax, Nmax, alpha)[:2]

    rdot = sph.RadialComponent(**kwargs)(m, Lmax, Nmax, alpha+1, exact=True)
    sdot = rdot[:,:2*sph.num_coeffs(Lmax, Nmax)]

    op = sdot @ sparse.vstack([gradp,gradm])
    op = sph.resize(op, Lmax+1, Nmax+1, Lmax, Nmax)

    return op.astype(dtype)


def Z(m, Lmax, Nmax, alpha, exact=False, dtype='float64', internal=internal_dtype):
    """Multiply a field by Z
       Codomain: (Lmax->Lmax+1, Nmax->Nmax+1, alpha->alpha) if exact
    """
    op = sph.RadialMultiplication()(m, Lmax, Nmax, alpha, exact=True)[2]

    Lout, Nout = (Lmax+1, Nmax+1) if exact else (Lmax, Nmax)
    op = sph.resize(op, Lmax+1, Nmax+1, Lout, Nout)

    return op.astype(dtype)


def Z_squared(m, Lmax, Nmax, alpha, exact=False, dtype='float64', internal=internal_dtype):
    """Multiply a field by Z**2
       Codomain (Lmax->Lmax+2, Nmax->Nmax+1, alpha->alpha) if exact
    """
    kwargs = {'dtype': internal, 'internal': internal}
    num_coeffs = sph.num_coeffs(Lmax+1, Nmax+1)

    rmul = sph.RadialMultiplication(**kwargs)
    zmul = rmul(m, Lmax, Nmax, alpha, exact=True)[2]

    rdot = sph.RadialComponent(**kwargs)
    zdot = rdot(m, Lmax+1, Nmax+1, alpha, exact=True)[:,2*num_coeffs:]

    op = zdot @ zmul

    Lout, Nout = (Lmax+2, Nmax+1) if exact else (Lmax, Nmax)
    op = sph.resize(op, Lmax+2, Nmax+2, Lout, Nout)

    return op.astype(dtype)


def ZdZ(m, Lmax, Nmax, alpha, dtype='float64', internal=internal_dtype):
    """z d/dz operator
       Codomain: (Lmax->Lmax, Nmax->Nmax, alpha->alpha+1)
    """
    kwargs = {'dtype': internal, 'internal': internal}
    num_coeffs = sph.num_coeffs(Lmax, Nmax)

    gradient = sph.Gradient(**kwargs)
    gradz = gradient(m, Lmax, Nmax, alpha)[2]

    rdot = sph.RadialComponent(**kwargs)(m, Lmax, Nmax, alpha+1, exact=True)
    zdot = rdot[:,2*num_coeffs:]

    op = zdot @ gradz
    op = sph.resize(op, Lmax+1, Nmax+1, Lmax, Nmax)

    return op.astype(dtype)


def dZ(m, Lmax, Nmax, alpha, dtype='float64', internal=internal_dtype):
    """d/dz operator
       Codomain: (Lmax->Lmax, Nmax->Nmax, alpha->alpha+1)
    """
    kwargs = {'dtype': internal, 'internal': internal}
    num_coeffs = sph.num_coeffs(Lmax, Nmax)

    gradient = sph.Gradient(**kwargs)
    op = gradient(m, Lmax, Nmax, alpha)[2]

    return op.astype(dtype)

