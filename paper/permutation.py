import numpy as np
import spherinder.operators as sph


def permutation_indices(Lmax, Nmax, galerkin, nfields):
    """For each mode interlace the field variables.  Returns two lists of
       permutation indices, the first for the columns (variable ordering), and
       the second for the rows (equation sorting).  Leaves tau variables as the
       final set of coefficients so the tau columns are in the same location -
       horizontally block appended to the matrix"""
    if isinstance(Lmax, int): Lmax = (Lmax,)*nfields
    if isinstance(Nmax, int): Nmax = (Nmax,)*nfields
    if isinstance(galerkin, bool): galerkin = (galerkin,)*nfields
    if len(Lmax) != nfields:
        raise ValueError("Lmax list size doesn't match nfields")
    if len(Nmax) != nfields:
        raise ValueError("Nmax list size doesn't match nfields")
    if len(galerkin) != nfields:
        raise ValueError("galerkin list size doesn't match nfields")

    var_lengths_offsets = [sph.coeff_sizes(L,          N)        for L,N   in zip(Lmax, Nmax)]
    eqn_lengths_offsets = [sph.coeff_sizes(L+2*int(g), N+int(g)) for L,N,g in zip(Lmax, Nmax, galerkin)]

    var_lengths = [sum(lengths_offsets[0]) for lengths_offsets in var_lengths_offsets]
    eqn_lengths = [sum(lengths_offsets[0]) for lengths_offsets in eqn_lengths_offsets]
    var_offsets = np.append(0, np.cumsum(var_lengths))
    eqn_offsets = np.append(0, np.cumsum(eqn_lengths))

    lengths_offsets, offsets = var_lengths_offsets, var_offsets
    varindices = []
    for ell in range(max(Lmax)):
        indices = []
        for i in range(nfields):
            if ell >= Lmax[i]: continue
            length, offset = lengths_offsets[i][0][ell], lengths_offsets[i][1][ell] + offsets[i]
            indices += list(range(offset, offset+length))
        varindices += np.ravel(indices).tolist()
    vartau = range(var_offsets[-1], eqn_offsets[-1])
    varindices = varindices + list(vartau)

    lengths_offsets, offsets = eqn_lengths_offsets, eqn_offsets
    eqnindices = []
    for ell in range(max(Lmax)+2):
        indices = []
        for i, is_galerkin in enumerate(galerkin):
            if ell >= Lmax[i]+2*int(is_galerkin): continue
            length, offset = lengths_offsets[i][0][ell], lengths_offsets[i][1][ell] + offsets[i]
            indices += list(range(offset, offset+length))
        eqnindices += np.ravel(indices).tolist()

    # Make sure we used each index once and only once
    neqn = sum(eqn_lengths)
    assert sorted(varindices) == list(range(neqn))
    assert sorted(eqnindices) == list(range(neqn))

    return varindices, eqnindices


def invert_permutation(permutation):
    """Invert a permutation"""
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv

