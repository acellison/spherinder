import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import dedalus.public as de

"""
Dedalus-provided solvers:
    ['UmfpackSpsolve', 'SuperluNaturalSpsolve', 'SuperluColamdSpsolve',
     'UmfpackFactorized', 'SuperluNaturalFactorized', 'SuperluColamdFactorized',
     'ScipyBanded', 'SPQR_solve', 'BandedQR', 'SparseInverse',
     'DenseInverse', 'BlockInverse']
Solvers defined in this module:
    ['LSQR_solve', 'UmfpackSpsolve64', 'UmfpackFactorized64', 
     'SuperluNaturalTransposed', 'SuperluColamdTransposed']
"""
matsolvers = de.matsolvers.matsolvers


def _to_int64(x, copy=True):
    """Force 64-bit indices so Umfpack doesn't raise an out of memory exception"""
    y = sparse.csc_matrix(x)
    if copy:
        y = y.copy()
    y.indptr = y.indptr.astype(np.int64)
    y.indices = y.indices.astype(np.int64)
    return y


SparseSolver = de.matsolvers.SparseSolver

# Add the LSQR solver to the dedalus matsolvers module
@de.matsolvers.add_solver
class LSQR_solve(SparseSolver):
    """LSQR solve."""

    def __init__(self, matrix, solver=None):
        self.matrix = matrix.copy()

    def solve(self, vector):
        n = np.prod(np.shape(vector))
        vector = np.reshape(np.asarray(vector.ravel()), n)
        solution = spla.lsqr(self.matrix, vector, atol=1e-14, btol=1e-14)
        return np.reshape(solution[0], (np.shape(self.matrix)[1], 1))


@de.matsolvers.add_solver
class UmfpackSpsolve64(SparseSolver):
    """UMFPACK spsolve, 64 bit indices."""

    def __init__(self, matrix, solver=None):
        from scikits import umfpack
        self.matrix = _to_int64(matrix, copy=True)

    def solve(self, vector):
        return spla.spsolve(self.matrix, vector, use_umfpack=True)


@de.matsolvers.add_solver
class UmfpackFactorized64(SparseSolver):
    """UMFPACK LU factorized solve, 64 bit indices."""

    def __init__(self, matrix, solver=None):
        from scikits import umfpack
        self.LU = spla.factorized(_to_int64(matrix, copy=True))

    def solve(self, vector):
        return self.LU(vector)


@de.matsolvers.add_solver
class SuperluNaturalTransposed(SparseSolver):
    """Super LU natural factorized transposed solve."""

    def __init__(self, matrix, solver=None):
        LU = spla.splu(matrix.T, permc_spec='NATURAL')
        self.L, self.U, self.perm_r, self.perm_c = LU.U.T, LU.L.T, LU.perm_r, LU.perm_c

    def solve(self, b):
        y = spla.spsolve_triangular(self.L, b[self.perm_c], lower=True,  unit_diagonal=False)
        x = spla.spsolve_triangular(self.U, y,              lower=False, unit_diagonal=True)
        return x[self.perm_r]


@de.matsolvers.add_solver
class SuperluColamdTransposed(SparseSolver):
    """Super LU colamd factorized transposed solve."""

    def __init__(self, matrix, solver=None):
        LU = spla.splu(matrix.T, permc_spec='COLAMD')
        self.L, self.U, self.perm_r, self.perm_c = LU.U.T, LU.L.T, LU.perm_r, LU.perm_c

    def solve(self, b):
        y = spla.spsolve_triangular(self.L, b[self.perm_c], lower=True,  unit_diagonal=False)
        x = spla.spsolve_triangular(self.U, y,              lower=False, unit_diagonal=True)
        return x[self.perm_r]


def make_solver(matsolver):
    """
    Construct a matrix solver object from a string.

    Parameters
    ----------
    matsolver : str
        String identifier of a matrix solver found in the matsolvers dict.
        If None, returns the default matsolver.
        If matsolver is instead a SparseSolver, returns it unmodified.
    """
    if matsolver is None:
        matsolver = 'SuperluNaturalSpsolve'

    if isinstance(matsolver, str):
        matsolver = matsolvers[matsolver.lower()]

    return matsolver

