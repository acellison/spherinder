import dedalus_sphere.jacobi as Jacobi
import numpy as np


def associated_legendre_nodes(n, m, dtype='float128', days=3):
    """Compute the nodes of the associated Legendre function P_n^{m}(z)"""
    z, _ = Jacobi.quadrature(n-m, m, m, dtype=dtype, days=days)
    return z


def compute_eigenvalues(n, m, dtype='float128', days=3):
    """Compute the eigenvalues of the inertial waves problem for a given
       degree n and azimuthal wavenumber m"""
    m = abs(m)
    if m == 0:
        N, a, b = n-m-1, m+1, m+1
    else:
        N, a, b = n-m, m-1, m+1
    z, _ = Jacobi.quadrature(N, a, b, dtype=dtype, days=days)
    return z


def compute_eigenmode(s, z, n, k, m):
    """Compute the eigenmode of the inertial waves problem for a given
       degree n and azimuthal wavenumber m with index k"""

    # Compute the eigenvalue
    roots = compute_eigenvalues(n, m)
    lam = roots[k-1]

    print('Greenspan Eigenvalue for mode ({},{},{}): {:1.6f}'.format(n,k,m,2*lam))
    
    # Compute the roots of the associated legendre function
    etaj = associated_legendre_nodes(n, m)
    etaj = np.sort(etaj)
    etaj = etaj[(1e-6 < etaj) & (etaj < 1-1e-6)]    

    # Construct the solution
    c = (1-lam**2)**(-1/2)
    enm = (n-m) % 2
    result = (lam*z)**enm * (s/c)**m
    for j in range(len(etaj)):
        eta2 = etaj[j]**2
        result *= (eta2*(1-lam**2)*s**2 + lam**2*(1-eta2)*z**2 + eta2*(eta2-1))
  
    result /= np.max(np.abs(result))
    return result


