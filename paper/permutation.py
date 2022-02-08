import numpy as np
import spherinder.operators as sph


def permutation_indices(nfields, Lmax, Nmax):
    """For each mode interlace the five field variables.  Returns two lists of
       permutation indices, the first for the columns (variable ordering), and
       the second for the rows (equation sorting).  Leaves tau variables as the
       final set of coefficients so the tau columns are in the same location -
       horizontally block appended to the matrix"""
    nvar = sph.num_coeffs(Lmax, Nmax)
    neqn = sph.num_coeffs(Lmax+2, Nmax+1)
    ntau = neqn - nvar

    lengths, offsets = sph.coeff_sizes(Lmax, Nmax)
    varindices = []
    for ell in range(Lmax):
        offset, length = offsets[ell], lengths[ell]
        variables = [list(range(offset+i*nvar, offset+i*nvar+length)) for i in range(nfields)]
        varindices += np.ravel(variables).tolist()
    vartau = range(nfields*nvar,nfields*neqn)
    varindices = varindices + list(vartau)

    lengths, offsets = sph.coeff_sizes(Lmax+2, Nmax+1)
    eqnindices = []
    for ell in range(Lmax+2):
        offset, length = offsets[ell], lengths[ell]
        equations = [list(range(offset+i*neqn, offset+i*neqn+length)) for i in range(nfields)]
        eqnindices += np.ravel(equations).tolist()

    return varindices, eqnindices


def invert_permutation(permutation):
    """Invert a permutation"""
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv

