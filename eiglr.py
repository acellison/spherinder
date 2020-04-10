from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import time

def mat2petsc(A):
    if A is None:
        return None

    return PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))


def eiglr(A, B=None, profile=False):
    """Compute the largest real eigenpair"""
    # Copy the matrices to PETSc format
    if profile:
        print("  Starting eigenvalue computation...")
        evals_start = time.time()

    n = np.shape(A)[0]
    A, B = mat2petsc(A), mat2petsc(B)

    # Create the eigenproblem 
    E = SLEPc.EPS(); E.create()                    # Create the eigensolver
    E.setOperators(A, B)                           # Setup the operators A v = k B v
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)  # Generalized eigenproblem
    E.setWhichEigenpairs(E.Which.LARGEST_REAL)     # Largest real eigenvalues
    E.setDimensions(nev=1)                         # Single eigenpair
    E.setFromOptions()

    # Solve the eigenproblem
    E.solve()

    if profile:
        evals_end = time.time()
        print("  Eigenvalues took {:g} sec".format(evals_end-evals_start))

    Print = PETSc.Sys.Print
    Print()
    Print("******************************")
    Print("*** SLEPc Solution Results ***")
    Print("******************************")
    Print()
    its = E.getIterationNumber()
    Print("Number of iterations of the method: %d" % its)
    eps_type = E.getType()
    Print("Solution method: %s" % eps_type)
    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %d" % nev)
    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    nconv = E.getConverged()
    Print("Number of converged eigenpairs %d" % nconv)

    if nconv > 0:
        # Create the results vectors
        vr, wr = A.getVecs()
        vi, wi = A.getVecs()

        # Get results
        lam = E.getEigenpair(0, vr, vi)
        v = np.zeros((n,1), dtype=np.complex128)
        v[:,0] = vr[:] + 1j*vi[:]

        return [lam], v
    else:
        raise ValueError('No converged results!')
        
