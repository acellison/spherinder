import dedalus_sphere.jacobi as Jacobi

# Jacobi operators
A = Jacobi.operator('A')
B = Jacobi.operator('B')
C = Jacobi.operator('C')
D = Jacobi.operator('D')
Z = Jacobi.operator('Z')
Id = Jacobi.operator('Id')


def compute_eigenvalues(n, m, maxiter=32):
    """Find the zeros of the equation: m * P_n^{|m|}(z) = (1-z**2) * d/dz P_n^{|m|}(z)"""
    Pm = lambda k, x: Jacobi.polynomials(n+2,k,k,x,dtype='float128').T
    op = m*(Id + Z) - A(-1) @ B(-1) @ D(+1)

    F = lambda x: (Pm(m, x) @ op(n+1,m,m))[:,n-m]
    Fp = lambda x: (Pm(m+1, x) @ (D(+1) @ op)(n+2,m,m))[:,n-m]

    z, _ = Jacobi.quadrature(2*(n+4), m, m, dtype='float128')
    for i in range(maxiter):
        Fz = F(z)
        if np.all(np.abs(Fz) < 1e-15):
            break
        z -= Fz/Fp(z)
    z = z[np.abs(z)<=1-1e-6]
    z = np.sort(z)
    z = np.append(z[0], z[1:][np.diff(z)>1e-12])

    return z


def associated_legendre_nodes(n, m, maxiter=3):
    z, _ = Jacobi.quadrature(n-m, m, m, dtype='float128', days=maxiter)
    return z


def compute_eigenmode(r, z, n, k, m):
    """Greenspan solution"""
    # Compute the eigenvalue
    roots = compute_eigenvalues(n, m)
    mu = roots[k-1]

    print('Greenspan Eigenvalue for mode ({},{},{}): {:1.6f}'.format(n,k,m,2*mu))
    
    # Compute the roots of the associated legendre function
    etaj = associated_legendre_nodes(n, m)
    etaj = np.sort(etaj)
    etaj = etaj[(1e-6 < etaj) & (etaj < 1-1e-6)]    

    # Construct the solution
    c = (1-mu**2)**(-1/2)
    enm = (n-m) % 2
    scale = 1
    result = scale * (mu*z)**enm * (r/c)**m
    for j in range(len(etaj)):
        eta = etaj[j]**2
        result = result * (eta*(1-mu**2)*r**2 + mu**2*(1-eta)*z**2 + eta*(eta-1))
  
    result /= np.max(np.abs(result))
    return result


