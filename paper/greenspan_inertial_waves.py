import numpy as np
import dedalus_sphere.jacobi as Jacobi
import spherinder.config as config
import spherinder.operators as sph


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


def compute_eigenmode(s, z, n, k, m, normalize=True, verbose=False):
    """Compute the eigenmode of the inertial waves problem for a given
       degree n and azimuthal wavenumber m with index k"""

    # Compute the eigenvalue
    roots = compute_eigenvalues(n, m)
    lam = roots[k-1]

    if verbose:
        print('Greenspan Eigenvalue for mode ({},{},{}): {:1.6f}'.format(n,k,m,2*lam))
    
    # Compute the roots of the associated legendre function
    etaj = associated_legendre_nodes(n, m)
    etaj = np.sort(etaj)
    tol = 1e-9
    etaj = etaj[(tol < etaj) & (etaj < 1-tol)]

    # Construct the solution
    c = (1-lam**2)**(-1/2)
    enm = (n-m) % 2
    result = (lam*z)**enm * (s/c)**m
    for j in range(len(etaj)):
        eta2 = etaj[j]**2
        result *= (eta2*(1-lam**2)*s**2 + lam**2*(1-eta2)*z**2 + eta2*(eta2-1))
  
    if normalize:
        result /= np.max(np.abs(result))
    return result


def project(f, m, Lmax, Nmax, alpha, sigma, dtype='float128', Lquad=None, Nquad=None, truncate=config.default_truncate):
    ratio = 10
    if Lquad is None:
        Lquad = ratio*Lmax
    if Nquad is None:
        Nquad = ratio*Nmax
    coeffs = np.zeros((Lmax, Nmax), dtype=dtype)
    eta, weta = Jacobi.quadrature(Lquad, alpha, alpha, dtype=dtype)
    Peta = Jacobi.polynomials(Lmax, alpha, alpha, eta, dtype=dtype)
    weta = weta[:,np.newaxis]
    Nsizes = sph.Nsizes(Lmax, Nmax, truncate=truncate)
    for ell in range(Lmax):
        N = Nsizes[ell]
        t, wt = Jacobi.quadrature(Nquad, ell+alpha+1/2, m+sigma, dtype=dtype)
        Pt = (1+t)**(-(m+sigma)/2) * (1-t)**(-ell/2) * Jacobi.polynomials(N, ell+alpha+1/2, m+sigma, t, dtype=dtype)
        feval = f(t,eta)
        coeffs[ell,:N] = [np.sum(weta * wt[np.newaxis,:] * Peta[ell][:,np.newaxis] * Pt[k] * feval) for k in range(N)]
    return coeffs


def reconstruct(coeffs, t, eta, m, Lmax, Nmax, alpha, sigma, dtype='float128'):
    Peta = Jacobi.polynomials(Lmax, alpha, alpha, eta, dtype=dtype)
    f = 0 * t[np.newaxis,:] * eta[:,np.newaxis]
    for ell in range(Lmax):
        Pt = (1+t)**((m+sigma)/2) * (1-t)**(ell/2) * Jacobi.polynomials(Nmax, ell+alpha+1/2, m+sigma, t, dtype=dtype)
        for k in range(Nmax):
            f += coeffs[ell,k] * Peta[ell][:,np.newaxis] * Pt[k][np.newaxis,:]
    return f


