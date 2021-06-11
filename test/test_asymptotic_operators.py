import spherinder.operators as sph
from spherinder.operators import Basis, plotfield
import spherinder.asymptotic_operators as sao

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


# Construct the spatial domain and bases
Lmax, Nmax, m = 6, 5, 2
ns, neta = 2000, 2001
alpha = 0

s, eta = np.linspace(0,1,ns), np.linspace(-1,1,neta)
ss, ee = s[np.newaxis,:], eta[:,np.newaxis]

basis = Basis(s, eta, m, Lmax, Nmax, sigma=0, alpha=alpha)

# Construct the coefficients
np.random.seed(seed=42)
c = np.random.random(basis.ncoeffs)
    
# Expand the function
f = basis.expand(c)

 
def check_close(a, b, tol, verbose=True, title=''):
    error = np.max(np.abs(a-b))
    if verbose:
        print(f'{title} L-Infinity error = {error}')
    assert error <= tol
    

def test_S_squared():
    cobasis = Basis(s, eta, m, Lmax, Nmax+1, sigma=0, alpha=alpha)
    
    # Apply the operator in grid space
    fgrid = ss**2 * f
    
    # Apply the operator in coefficient space
    op = sao.S_squared(m, Lmax, Nmax, alpha, exact=True)
    
    # Apply the operator
    d = op @ c.ravel()
    fcoeff = cobasis.expand(d)

    check_close(fgrid, fcoeff, 4e-14, title='S**2')


def test_S_fourth():
    cobasis = Basis(s, eta, m, Lmax, Nmax+2, sigma=0, alpha=alpha)
    
    # Apply the operator in grid space
    fgrid = ss**4 * f
    
    # Apply the operator in coefficient space
    op = sao.S_fourth(m, Lmax, Nmax, alpha, exact=True)
    
    # Apply the operator
    d = op @ c.ravel()
    fcoeff = cobasis.expand(d)

    check_close(fgrid, fcoeff, 4e-14, title='S**4')



def test_Z():
    cobasis = Basis(s, eta, m, Lmax+1, Nmax+1, sigma=0, alpha=alpha)

    # Apply the operator in grid space
    fgrid = ee * np.sqrt(1-ss**2) * f
    
    # Apply the operator in coefficient space
    op = sao.Z(m, Lmax, Nmax, alpha, exact=True)
    
    # Apply the operator
    d = op @ c.ravel()
    fcoeff = cobasis.expand(d)

    check_close(fgrid, fcoeff, 4e-14, title='Z')


def test_Z_squared():
    cobasis = Basis(s, eta, m, Lmax+2, Nmax+1, sigma=0, alpha=alpha)

    # Apply the operator in grid space
    fgrid = (ee * np.sqrt(1-ss**2))**2 * f
    
    # Apply the operator in coefficient space
    op = sao.Z_squared(m, Lmax, Nmax, alpha, exact=True)
    
    # Apply the operator
    d = op @ c.ravel()
    fcoeff = cobasis.expand(d)

    check_close(fgrid, fcoeff, 4e-14, title='Z**2')



def test_SdS():
    cobasis = Basis(s, eta, m, Lmax, Nmax, sigma=0, alpha=alpha+1)
    
    # Apply the operator in grid space
    dfds = (f[:,2:]-f[:,:-2])/(ss[:,2:]-ss[:,:-2])
    
    scale = 1-ss**2
    scale[np.abs(scale) < 1e-12] = 1
    dfdeta = (ss/scale)*ee[1:-1,:]*(f[2:,:]-f[:-2,:])/(ee[2:,:]-ee[:-2,:])
    
    fgrid = ss[:,1:-1] * (dfds[1:-1,:] + dfdeta[:,1:-1])
    
    # Apply the operator in coefficient space
    op = sao.SdS(m, Lmax, Nmax, alpha)
    
    # Apply the operator
    d = op @ c.ravel()
    fcoeff = cobasis.expand(d)
    fcoeff = fcoeff[1:-1,1:-1]
    
    send = np.argmin(s[1:]<.95)
    fgrid, fcoeff = fgrid[:,:send], fcoeff[:,:send]
    
    check_close(fgrid, fcoeff, 0.01, title='SdS')

    
def test_dZ():
    cobasis = Basis(s, eta, m, Lmax, Nmax, sigma=0, alpha=alpha+1)

    # Apply the operator in grid space
    scale = np.sqrt(1-ss**2)
    scale[np.abs(scale) < 1e-12] = 1
    fgrid = (1/scale)*(f[2:,:]-f[:-2,:])/(ee[2:,:]-ee[:-2,:])
    
    # Apply the operator in coefficient space
    op = sao.dZ(m, Lmax, Nmax, alpha)
    
    # Apply the operator
    d = op @ c.ravel()
    fcoeff = cobasis.expand(d)[1:-1,:]

    send = np.argmin(s[1:]<.95)
    fgrid, fcoeff = fgrid[:,:send], fcoeff[:,:send]

    check_close(fgrid, fcoeff, 0.001, title='dZ')


def test_ZdZ():
    cobasis = Basis(s, eta, m, Lmax, Nmax, sigma=0, alpha=alpha+1)

    # Apply the operator in grid space
    fgrid = ee[1:-1,:] * (f[2:,:]-f[:-2,:])/(ee[2:,:]-ee[:-2,:])
    
    # Apply the operator in coefficient space
    op = sao.ZdZ(m, Lmax, Nmax, alpha)
    
    # Apply the operator
    d = op @ c.ravel()
    fcoeff = cobasis.expand(d)[1:-1,:]

    send = np.argmin(s[1:]<.95)
    fgrid, fcoeff = fgrid[:,:send], fcoeff[:,:send]

    check_close(fgrid, fcoeff, 0.001, title='ZdZ')


def main():
    print('Testing...')

    test_Z()
    test_Z_squared()
    test_dZ()
    test_ZdZ()
    test_S_squared()
    test_S_fourth()
    test_SdS()

    print('All tests passed')


if __name__=='__main__':
    main()

