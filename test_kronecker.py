import dedalus_sphere.zernike as Zernike
import dedalus_sphere.jacobi as Jacobi
from kronecker import Kronecker

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

dim = 2

# Jacobi operators
A = Jacobi.operator('A')
B = Jacobi.operator('B')
C = Jacobi.operator('C')
DZ = Jacobi.operator('D')
Z = Jacobi.operator('Z')
IdZ = Jacobi.operator('Id')

# Zernike operators
DR = Zernike.operator(dim, 'D')
E = Zernike.operator(dim, 'E')
R = Zernike.operator(dim, 'R')
IdR = Zernike.operator(dim, 'Id')

# Composite Jacobi operators
Z2 = Z**2                                     # Z**2
ZdZ = Z @ DZ(+1)                              # Z * d/dZ
AB = A(+1) @ B(+1)                            # a,b -> a+1,b+1

# Composite Zernike operators
R2 = R(-1) @ R(+1)                            # R**2
H = IdR - R2                                  # 1 - R**2
H2 = H**2                                     # (1 - R**2)**2
RdR = (R(+1) @ DR(-1) + R(-1) @ DR(+1))/2     # R * d/dR


def make_square_resizer(dimensions):
    def resizer(dim, mat):
        n = dimensions[dim]
        mat = mat[:n, :n]
        ncols = np.shape(mat)[1]
        if ncols < n:
            mat = sparse.hstack([mat, np.zeros((n, n - ncols))])
        return mat
    return resizer


def main():
    # Setup the domain
    m, Nmax, Lmax = 3, 100, 10
    a, b, k = -1/2, -1/2, 0

    resizer = make_square_resizer([Nmax, Lmax])

    # Construct the composite operator [1 - S**2 + S**2*eta**2]*(d/deta)**2
    ops1 = H
    opz1 = IdZ
    op1 = Kronecker([ops1, opz1], resizer=resizer)

    ops2 = R2
    opz2 = Z2
    op2 = Kronecker([ops2, opz2], resizer=resizer)

    ops3 = IdR
    opz3 = DZ(+1)**2
    op3 = Kronecker([ops3, opz3], resizer=resizer)

    op = (op1 + op2) @ op3
    mat = op([(Nmax,k,m),(Lmax,a,b)])

    # Construct the operator by hand
    ops1 = H
    opz1 = DZ(+1)**2
    ops2 = R2
    opz2 = Z2 @ DZ(+1)**2
    mat1 = sparse.kron(resizer(0,ops1(Nmax,k,m)),resizer(1,opz1(Lmax,a,b)))
    mat2 = sparse.kron(resizer(0,ops2(Nmax,k,m)),resizer(1,opz2(Lmax,a,b)))
    mat_hand = mat1 + mat2

    assert np.max(abs(mat - mat_hand)) < 1.5e-14


if __name__=='__main__':
    main()
