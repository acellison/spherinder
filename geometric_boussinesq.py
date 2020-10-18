import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import os
import pickle

from eigtools import eigsort, scipy_sparse_eigs, plot_spectrum, discard_spurious_eigenvalues
import spherinder as sph

g_alpha_p = 2
g_alpha_T = 0
g_file_prefix = 'geometric_linear_onset'


def matrices_tau(m, Lmax, Nmax, Ekman, Prandtl, Rayleigh):
    """Construct matrices for X = [u(+), u(-), u(z), p, T]
    """
    alpha_bc, alpha_bc_s, alpha_bc_T = 2, 1, g_alpha_T+1
    ncoeff = Lmax*Nmax
    Zero = sparse.lil_matrix((ncoeff,ncoeff))
    I = sparse.eye(ncoeff)

    # Scalar laplacian
    LapT = sph.operator('lap')(m, Lmax, Nmax, alpha=g_alpha_T)
    LapT = sph.resize(LapT, Lmax, Nmax+1, Lmax, Nmax)

    # Vector laplacian
    Lapp, Lapm, Lapz = sph.operator('lap', 'vector')(m, Lmax, Nmax, alpha=1)
    Lapp = sph.resize(Lapp, Lmax, Nmax+1, Lmax, Nmax)
    Lapm = sph.resize(Lapm, Lmax, Nmax+1, Lmax, Nmax)
    Lapz = sph.resize(Lapz, Lmax, Nmax+1, Lmax, Nmax)

    # Vector divergence operator
    Div = sph.operator('div')(m, Lmax, Nmax, alpha=1)
    Div = sph.resize(Div, Lmax, Nmax+1, Lmax, Nmax)      # truncate Div . e(+)^* . u
    Divp, Divm, Divz = Div[:,:ncoeff], Div[:,ncoeff:2*ncoeff], Div[:,2*ncoeff:]

    # Convert operators to the proper alpha spaces
    # u(+) conversion from alpha=1 to alpha=3
    Cp = sph.convert_alpha(2, m, Lmax, Nmax, alpha=1, sigma=+1, truncate=True)

    # u(-) conversion from alpha=1 to alpha=3
    Cm = sph.convert_alpha(2, m, Lmax, Nmax, alpha=1, sigma=-1, truncate=True)

    # u(z) conversion from alpha=1 to alpha=3
    Cz = sph.convert_alpha(2, m, Lmax, Nmax, alpha=1, sigma=0, truncate=True)
 
    # scalar conversion from alpha to alpha+2
    Cs = sph.convert_alpha(2, m, Lmax, Nmax, alpha=g_alpha_T, sigma=0, truncate=True)

    # Pressure gradient
    Gradp, Gradm, Gradz = sph.operator('grad')(m, Lmax, Nmax, alpha=g_alpha_p)
    Cgp = sph.convert_alpha(2-g_alpha_p, m, Lmax,   Nmax,   alpha=1+g_alpha_p, sigma=+1, truncate=False)
    Cgm = sph.convert_alpha(2-g_alpha_p, m, Lmax,   Nmax+1, alpha=1+g_alpha_p, sigma=-1, truncate=False)
    Cgz = sph.convert_alpha(2-g_alpha_p, m, Lmax-1, Nmax,   alpha=1+g_alpha_p, sigma=0,  truncate=False)
    Gradp, Gradm, Gradz = Cgp @ Gradp, Cgm @ Gradm, Cgz @ Gradz
    Gradp = sph.resize(Gradp, Lmax,   Nmax+1, Lmax, Nmax)
    Gradm = sph.resize(Gradm, Lmax,   Nmax+2, Lmax, Nmax)
    Gradz = sph.resize(Gradz, Lmax-1, Nmax+1, Lmax, Nmax)

    # Radial vector extraction
    Rad = sph.operator('erdot')(m, Lmax, Nmax, alpha=1)
    Cr = sph.convert_alpha(1+g_alpha_T, m, Lmax+1, Nmax+1, alpha=1, sigma=0, truncate=False)
    Rad = Cr @ Rad
    Rad = sph.resize(Rad, Lmax+1, Nmax+2, Lmax, Nmax)
    Radp, Radm, Radz = Rad[:,:ncoeff], Rad[:,ncoeff:2*ncoeff], Rad[:,2*ncoeff:]

    # Radial vector multiplication r e_r * T, convert from alpha=1 to alpha=3
    RTp, RTm, RTz = sph.operator('rtimes')(m, Lmax, Nmax, alpha=g_alpha_T)
    CrTp = sph.convert_alpha(2-g_alpha_T, m, Lmax,   Nmax+1, alpha=1+g_alpha_T, sigma=+1, truncate=False)
    CrTm = sph.convert_alpha(2-g_alpha_T, m, Lmax,   Nmax+2, alpha=1+g_alpha_T, sigma=-1, truncate=False)
    CrTz = sph.convert_alpha(2-g_alpha_T, m, Lmax+1, Nmax+2, alpha=1+g_alpha_T, sigma=0,  truncate=False)
    RTp, RTm, RTz = CrTp @ RTp, CrTm @ RTm, CrTz @ RTz
    RTp = sph.resize(RTp, Lmax,   Nmax+2, Lmax, Nmax)
    RTm = sph.resize(RTm, Lmax,   Nmax+3, Lmax, Nmax)
    RTz = sph.resize(RTz, Lmax+1, Nmax+3, Lmax, Nmax)

    # Boundary operator
    Boundary = sph.operator('boundary')
    Boundp = Boundary(m, Lmax, Nmax, alpha=1, sigma=+1)
    Boundm = Boundary(m, Lmax, Nmax, alpha=1, sigma=-1)
    Boundz = Boundary(m, Lmax, Nmax, alpha=1, sigma=0)
    BoundT = Boundary(m, Lmax, Nmax, alpha=g_alpha_T, sigma=0)
    Bound = sparse.bmat([[  Boundp, 0*Boundm, 0*Boundz, 0*BoundT, 0*BoundT],
                         [0*Boundp,   Boundm, 0*Boundz, 0*BoundT, 0*BoundT],
                         [0*Boundp, 0*Boundm,   Boundz, 0*BoundT, 0*BoundT],
                         [0*Boundp, 0*Boundm, 0*Boundz, 0*BoundT,   BoundT]])

    # Time derivative matrices
    M00 = Ekman * Cp
    M11 = Ekman * Cm
    M22 = Ekman * Cz
    M33 = Zero
    M44 = Prandtl * Cs

    # i*u+ equation - spin+ velocity component
    L00 = -1j * Cp + Ekman * Lapp
    L01 = Zero
    L02 = Zero
    L03 = -Gradp
    L04 = Rayleigh * RTp

    # i*u- equation - spin- velocity component
    L10 = Zero
    L11 = 1j * Cm + Ekman * Lapm
    L12 = Zero
    L13 = -Gradm
    L14 = Rayleigh * RTm

    # i*w equation - vertical velocity component
    L20 = Zero
    L21 = Zero
    L22 = Ekman * Lapz
    L23 = -Gradz
    L24 = Rayleigh * RTz

    # Divergence equation
    L30 = Divp
    L31 = Divm
    L32 = Divz
    L33 = Zero
    L34 = Zero

    # Temperature equation
    L40 = Prandtl * Radp
    L41 = Prandtl * Radm
    L42 = Prandtl * Radz
    L43 = Zero
    L44 = LapT

    Mmats = [M00, M11, M22, M33, M44]
    upmats = [L00, L01, L02, L03, L04]
    ummats = [L10, L11, L12, L13, L14]
    uzmats = [L20, L21, L22, L23, L24]
    pmats = [L30, L31, L32, L33, L34]
    Tmats = [L40, L41, L42, L43, L44]

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([upmats, ummats, uzmats, pmats, Tmats], format=sparse_format)

    # Boundary conditions
    def no_slip():
        row = Bound

        Taup = sph.convert_alpha(3-alpha_bc, m, Lmax, Nmax, alpha=alpha_bc, sigma=+1, truncate=True)
        Taum = sph.convert_alpha(3-alpha_bc, m, Lmax, Nmax, alpha=alpha_bc, sigma=-1, truncate=True)
        Tauz = sph.convert_alpha(3-alpha_bc, m, Lmax, Nmax, alpha=alpha_bc, sigma=0, truncate=True)
        Taus = sph.convert_alpha(2-alpha_bc_s, m, Lmax, Nmax, alpha=alpha_bc_s, sigma=0, truncate=True)
        TauT = sph.convert_alpha(2+g_alpha_T-alpha_bc_T, m, Lmax, Nmax, alpha=alpha_bc_T, sigma=0, truncate=True)
        taup, taum, tauz, taus, tauT = Taup[:,-2*Nmax:], Taum[:,-2*Nmax:], Tauz[:,-2*Nmax:], Taus[:,-2*Nmax:], TauT[:,-2*Nmax:]
        col1 = sparse.bmat([[  taup,0*taum,0*tauz,0*tauT],
                            [0*taup,  taum,0*tauz,0*tauT],
                            [0*taup,0*taum,  tauz,0*tauT],
                            [0*taup,0*taum,0*tauz,0*tauT],
                            [0*taup,0*taum,0*tauz,  tauT]])

        taup, taum, tauz, taus, tauT = Taup[:,Nmax-1:-2*Nmax:Nmax], Taum[:,Nmax-1:-2*Nmax:Nmax], \
                                       Tauz[:,Nmax-1:-2*Nmax:Nmax], Taus[:,Nmax-1:-2*Nmax:Nmax], \
                                       TauT[:,Nmax-1:-2*Nmax:Nmax]
        col2 = sparse.bmat([[  taup,0*taum,0*tauz,0*tauT],
                            [0*taup,  taum,0*tauz,0*tauT],
                            [0*taup,0*taum,  tauz,0*tauT],
                            [0*taup,0*taum,0*tauz,0*tauT],
                            [0*taup,0*taum,0*tauz,  tauT]])
 
        col = sparse.hstack([col1,col2])
        return row, col

    # Create the boundary condition rows and tau columns
    row, col = no_slip()

    corner = np.zeros((np.shape(row)[0], np.shape(col)[1]))
    L = sparse.bmat([[  L, col],
                     [row, corner]], format='csr')

    M = sparse.bmat([[    M, 0*col],
                     [0*row, corner]], format='csr')

    M, L = M.tocsr(), L.tocsr()
    return M, L


def matrices_galerkin(m, Lmax, Nmax, Ekman, Prandtl, Rayleigh):
    """Construct matrices for X = [u(+), u(-), u(z), p, T]
    """
    alpha_bc, alpha_bc_s, alpha_bc_T = 2, 1, g_alpha_T+1

    Lout, Nout = Lmax+2, Nmax+1
    ncoeff = Lout*Nout
    ncoeff0 = Lmax*Nmax

    # Galerkin conversion operators
    Boundp = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=+1)
    Boundm = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=-1)
    Boundz = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=2, sigma=0)
    BoundT = sph.operator('1-r**2')(m, Lmax, Nmax, alpha=1+g_alpha_T, sigma=0)

    # Scalar laplacian
    LapT = sph.operator('lap')(m, Lout, Nout, alpha=g_alpha_T)
    LapT = sph.resize(LapT, Lout, Nout+1, Lout, Nout)

    # Vector laplacian
    Lapp, Lapm, Lapz = sph.operator('lap', 'vec')(m, Lout, Nout, alpha=1)
    Lapp = sph.resize(Lapp, Lout, Nout+1, Lout, Nout)
    Lapm = sph.resize(Lapm, Lout, Nout+1, Lout, Nout)
    Lapz = sph.resize(Lapz, Lout, Nout+1, Lout, Nout)

    # Vector divergence operator
    Div = sph.operator('div')(m, Lout, Nout, alpha=1)
    Div = sph.resize(Div, Lout, Nout+1, Lout, Nout)      # truncate Div . e(+)^* . u
    Divp, Divm, Divz = Div[:,:ncoeff], Div[:,ncoeff:2*ncoeff], Div[:,2*ncoeff:]

    # Pressure gradient
    Gradp, Gradm, Gradz = sph.operator('grad')(m, Lmax, Nmax, alpha=g_alpha_p)
    Cgp = sph.convert_alpha(2-g_alpha_p, m, Lmax,   Nmax,   alpha=1+g_alpha_p, sigma=+1, truncate=False)
    Cgm = sph.convert_alpha(2-g_alpha_p, m, Lmax,   Nmax+1, alpha=1+g_alpha_p, sigma=-1, truncate=False)
    Cgz = sph.convert_alpha(2-g_alpha_p, m, Lmax-1, Nmax,   alpha=1+g_alpha_p, sigma=0,  truncate=False)
    Gradp, Gradm, Gradz = Cgp @ Gradp, Cgm @ Gradm, Cgz @ Gradz
    Gradp = sph.resize(Gradp, Lmax,   Nmax+1, Lout, Nout)
    Gradm = sph.resize(Gradm, Lmax,   Nmax+2, Lout, Nout)
    Gradz = sph.resize(Gradz, Lmax-1, Nmax+1, Lout, Nout)

    # Radial vector extraction
    Rad = sph.operator('erdot')(m, Lout, Nout, alpha=1)
    Cr = sph.convert_alpha(1+g_alpha_T, m, Lout+1, Nout+1, alpha=1, sigma=0, truncate=False)
    Rad = Cr @ Rad
    Rad = sph.resize(Rad, Lout+1, Nout+2, Lout, Nout)
    Radp, Radm, Radz = Rad[:,:ncoeff], Rad[:,ncoeff:2*ncoeff], Rad[:,2*ncoeff:]

    # Radial vector multiplication r e_r * T, convert from alpha=1 to alpha=3
    RTp, RTm, RTz = sph.operator('rtimes')(m, Lout, Nout, alpha=g_alpha_T)
    CrTp = sph.convert_alpha(2-g_alpha_T, m, Lout,   Nout+1, alpha=1+g_alpha_T, sigma=+1, truncate=False)
    CrTm = sph.convert_alpha(2-g_alpha_T, m, Lout,   Nout+2, alpha=1+g_alpha_T, sigma=-1, truncate=False)
    CrTz = sph.convert_alpha(2-g_alpha_T, m, Lout+1, Nout+2, alpha=1+g_alpha_T, sigma=0,  truncate=False)
    RTp, RTm, RTz = CrTp @ RTp, CrTm @ RTm, CrTz @ RTz
    RTp = sph.resize(RTp, Lout,   Nout+2, Lout, Nout)
    RTm = sph.resize(RTm, Lout,   Nout+3, Lout, Nout)
    RTz = sph.resize(RTz, Lout+1, Nout+3, Lout, Nout)

    # Conversion matrices
    Cp = sph.convert_alpha(2, m, Lout, Nout, alpha=1, sigma=+1, truncate=True)
    Cm = sph.convert_alpha(2, m, Lout, Nout, alpha=1, sigma=-1, truncate=True)
    Cz = sph.convert_alpha(2, m, Lout, Nout, alpha=1, sigma=0, truncate=True)
    CT = sph.convert_alpha(2, m, Lout, Nout, alpha=g_alpha_T, sigma=0, truncate=True)
    
    # Time derivative matrices
    M00 = Ekman * Cp @ Boundp
    M11 = Ekman * Cm @ Boundm
    M22 = Ekman * Cz @ Boundz
    M33 = sparse.lil_matrix((ncoeff,ncoeff0))
    M44 = Prandtl * CT @ BoundT

    # i*u+ equation - spin+ velocity component
    L00 = (-1j * Cp + Ekman * Lapp) @ Boundp
    L01 = sparse.lil_matrix((ncoeff,ncoeff0))
    L02 = sparse.lil_matrix((ncoeff,ncoeff0))
    L03 = -Gradp
    L04 = Rayleigh * RTp @ BoundT

    # i*u- equation - spin- velocity component
    L10 = sparse.lil_matrix((ncoeff,ncoeff0))
    L11 = (1j * Cm + Ekman * Lapm) @ Boundm
    L12 = sparse.lil_matrix((ncoeff,ncoeff0))
    L13 = -Gradm
    L14 = Rayleigh * RTm @ BoundT

    # i*w equation - vertical velocity component
    L20 = sparse.lil_matrix((ncoeff,ncoeff0))
    L21 = sparse.lil_matrix((ncoeff,ncoeff0))
    L22 = Ekman * Lapz @ Boundz
    L23 = -Gradz
    L24 = Rayleigh * RTz @ BoundT

    # Divergence equation
    L30 = Divp @ Boundp
    L31 = Divm @ Boundm
    L32 = Divz @ Boundz
    L33 = sparse.lil_matrix((ncoeff,ncoeff0))
    L34 = sparse.lil_matrix((ncoeff,ncoeff0))

    # Temperature equation
    L40 = Prandtl * Radp @ Boundp
    L41 = Prandtl * Radm @ Boundm
    L42 = Prandtl * Radz @ Boundz
    L43 = sparse.lil_matrix((ncoeff,ncoeff0))
    L44 = LapT @ BoundT

    Mmats = [M00, M11, M22, M33, M44]
    upmats = [L00, L01, L02, L03, L04]
    ummats = [L10, L11, L12, L13, L14]
    uzmats = [L20, L21, L22, L23, L24]
    pmats = [L30, L31, L32, L33, L34]
    Tmats = [L40, L41, L42, L43, L44]

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([upmats, ummats, uzmats, pmats, Tmats], format=sparse_format)

    # Tau polynomials
    def tau_polynomials():
        def make_tau_column(a,b,c,d,e):
            return sparse.bmat([[  a,0*b,0*c,0*d,0*e],
                                [0*a,  b,0*c,0*d,0*e],
                                [0*a,0*b,  c,0*d,0*e],
                                [0*a,0*b,0*c,  d,0*e],
                                [0*a,0*b,0*c,0*d,  e]])
        hstack = sparse.hstack

        Taup = sph.convert_alpha(3-alpha_bc, m, Lout, Nout, alpha=alpha_bc, sigma=+1, truncate=True)
        Taum = sph.convert_alpha(3-alpha_bc, m, Lout, Nout, alpha=alpha_bc, sigma=-1, truncate=True)
        Tauz = sph.convert_alpha(3-alpha_bc, m, Lout, Nout, alpha=alpha_bc, sigma=0, truncate=True)
        Taus = sph.convert_alpha(2-alpha_bc_s, m, Lout, Nout, alpha=alpha_bc_s, sigma=0, truncate=True)
        TauT = sph.convert_alpha(2+g_alpha_T-alpha_bc_T, m, Lout, Nout, alpha=alpha_bc_T, sigma=0, truncate=True)

        taup1, taum1, tauz1, taus1, tauT1 = Taup[:,Nout-1:-2*Nout:Nout], Taum[:,Nout-1:-2*Nout:Nout], Tauz[:,Nout-1:-2*Nout:Nout], \
                                            Taus[:,Nout-1:-2*Nout:Nout], TauT[:,Nout-1:-2*Nout:Nout]
        taup2, taum2, tauz2, taus2, tauT2 = Taup[:,-2*Nout:], Taum[:,-2*Nout:], Tauz[:,-2*Nout:], \
                                            Taus[:,-2*Nout:], TauT[:,-2*Nout:]
 
        col = make_tau_column(hstack([taup1,taup2]), hstack([taum1,taum2]), hstack([tauz1,tauz2]), \
                              hstack([taus1,taus2]), hstack([tauT1,tauT2]))
        return col

    col = tau_polynomials()

    L = sparse.hstack([L,  col], format='csr')
    M = sparse.hstack([M,0*col], format='csr')

    return M, L


def matrices(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh):
    if boundary_method in ['tau', 'galerkin']:
        return eval('matrices_' + boundary_method)(m, Lmax, Nmax, Ekman, Prandtl, Rayleigh)
    else:
        raise ValueError('Unsupported boundary method')


def permutation_indices(Lmax, Nmax):
    """For each mode interlace the five field variables.  Returns two lists of
       permutation indices, the first for the columns (variable ordering), and
       the second for the rows (equation sorting).  Leaves tau variables as the
       final set of coefficients so the tau columns are in the same location -
       horizontally block appended to the matrix"""
    nfields = 5
    nvar = Lmax*Nmax
    neqn = (Lmax+2)*(Nmax+1)
    ntau = 2*(Nmax+1)+Lmax

    variables = [range(i*nvar,(i+1)*nvar) for i in range(nfields)]
    equations = [range(i*neqn,(i+1)*neqn) for i in range(nfields)]

    vartau = range(nfields*nvar,nfields*(nvar+ntau))
    varindices = [val for tup in zip(*variables) for val in tup]
    varindices = varindices + list(vartau)
    eqnindices = [val for tup in zip(*equations) for val in tup]
    return varindices, eqnindices


def invert_permutation(permutation):
    """Invert a permutation"""
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv


def checkdir(filename):
    filename = os.path.abspath(filename)
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.mkdir(path)

def savedata(filename, data):
    checkdir(filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def savefig(filename):
    checkdir(filename)
    plt.savefig(filename)
    

def make_filename_prefix(directory='data'):
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), directory))
    abspath = os.path.join(basepath, g_file_prefix)
    if not os.path.exists(abspath):
        os.makedirs(abspath)
    return os.path.join(abspath, g_file_prefix)


def pickle_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, directory='data'):
    return make_filename_prefix(directory) + f'-evalues-m={m}-Lmax={Lmax}-Nmax={Nmax}-Ekman={Ekman:1.4e}-Prandtl={Prandtl}-Rayleigh={Rayleigh:1.4e}-{boundary_method}.pckl'


def solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, omega, plot_spy, nev='all'):
    # Construct the system
    print('Constructing matrix system...')
    M, L = matrices(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh)

    permute = True
    enable_permutation = permute and boundary_method == 'galerkin'

    if enable_permutation:
        print('Reordering variables and equations...')
        var, eqn = permutation_indices(Lmax, Nmax)
        M, L = M[:,var], L[:,var]
        M, L = M[eqn,:], L[eqn,:]

    if plot_spy:
        fig, plot_axes = plt.subplots(1,2,figsize=(9,4))
        plot_axes[0].spy(L)
        plot_axes[1].spy(M)
        plt.show()

    # Compute the eigenvalues and eigenvectors
    print('Eigenvalue problem, size {}'.format(np.shape(L)))
    if nev == 'all':
        evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)
    else:
        evalues, evectors = scipy_sparse_eigs(L, M, nev, target=omega, profile=True)

    if enable_permutation:
        vari = invert_permutation(var)
        evectors = evectors[vari,:]

    # Output data
    data = {'m': m, 'Lmax': Lmax, 'Nmax': Nmax, 
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors,
            'Ekman': Ekman, 'Prandtl': Prandtl, 'Rayleigh': Rayleigh}
    filename = pickle_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh)
    savedata(filename, data)


def create_bases(m, Lmax, Nmax, boundary_method, s, eta):
    if boundary_method == 'galerkin':
        basis_fun = sph.phi
        dalpha = 1
    else:
        basis_fun = sph.psi
        dalpha = 0

    upbasis = [basis_fun(Nmax, m, ell, s, eta, sigma=+1, alpha=1+dalpha) for ell in range(Lmax)]
    umbasis = [basis_fun(Nmax, m, ell, s, eta, sigma=-1, alpha=1+dalpha) for ell in range(Lmax)]
    uzbasis = [basis_fun(Nmax, m, ell, s, eta, sigma= 0, alpha=1+dalpha) for ell in range(Lmax)]
    pbasis  = [  sph.psi(Nmax, m, ell, s, eta, sigma= 0, alpha=g_alpha_p) for ell in range(Lmax)]
    Tbasis  = [basis_fun(Nmax, m, ell, s, eta, sigma= 0, alpha=g_alpha_T+dalpha) for ell in range(Lmax)]
    bases = {'up':upbasis, 'um':umbasis, 'uz':uzbasis, 'p':pbasis, 'T':Tbasis}

    return bases


def expand_evectors(Lmax, Nmax, vec, bases, domain):
    ncoeff = Lmax*Nmax

    # Get the grid space vector fields
    upcoeff = vec[:ncoeff] 
    umcoeff = vec[ncoeff:2*ncoeff] 
    uzcoeff = vec[2*ncoeff:3*ncoeff]
    pcoeff = vec[3*ncoeff:4*ncoeff]
    Tcoeff = vec[4*ncoeff:5*ncoeff]
    tau = vec[5*ncoeff:]
    print('Tau norm: {}'.format(np.linalg.norm(tau)))

    # Convert to grid space
    if bases is not None:
        up = sph.expand(bases['up'], upcoeff.reshape((Lmax,Nmax)))
        um = sph.expand(bases['um'], umcoeff.reshape((Lmax,Nmax)))
        uz = sph.expand(bases['uz'], uzcoeff.reshape((Lmax,Nmax)))
        p  = sph.expand(bases['p'],   pcoeff.reshape((Lmax,Nmax)))
        T  = sph.expand(bases['T'],   Tcoeff.reshape((Lmax,Nmax)))
    else:
        m, Lmax, Nmax, s, eta, boundary_method = domain['m'], domain['Lmax'], domain['Nmax'], domain['s'], domain['eta'], domain['boundary_method']
        dalpha = 1 if boundary_method == 'galerkin' else 0
        up = sph.expand_low_storage(upcoeff.reshape((Lmax,Nmax)), m, s, eta, sigma=+1, alpha=1+dalpha, basis_kind=boundary_method)
        um = sph.expand_low_storage(umcoeff.reshape((Lmax,Nmax)), m, s, eta, sigma=-1, alpha=1+dalpha, basis_kind=boundary_method)
        uz = sph.expand_low_storage(uzcoeff.reshape((Lmax,Nmax)), m, s, eta, sigma= 0, alpha=1+dalpha, basis_kind=boundary_method)
        p  = sph.expand_low_storage( pcoeff.reshape((Lmax,Nmax)), m, s, eta, sigma= 0, alpha=g_alpha_p)
        T  = sph.expand_low_storage( Tcoeff.reshape((Lmax,Nmax)), m, s, eta, sigma= 0, alpha=g_alpha_T+dalpha, basis_kind=boundary_method)

    u, v, w = np.sqrt(0.5)*(up + um), -1j * np.sqrt(0.5)*(up - um), uz

    return u, v, w, p, T, tau


def plot_spectrum_callback(index, evalues, evectors, Lmax, Nmax, s, eta, bases, domain):
    evalue, evector = evalues[index], evectors[:,index]
    u, v, w, p, T, tau = expand_evectors(Lmax, Nmax, evector, bases, domain)

    field_indices = [0,2,3,4]
    fields = [u,v,w,p,T]
    field_names = ['u','v','w','p','T']

    fig, ax = plt.subplots(1,len(field_indices),figsize=(13,4.5))
    for i in range(len(field_indices)):
        field_index = field_indices[i]
        f = fields[field_index]
        sph.plotfield(s, eta, f.real, fig=fig, ax=ax[i])
        ax[i].set_title(r'${}$'.format(field_names[field_index]))

        if field_index in [0,1,2,4]:
            error = max(np.linalg.norm(f[0,:]), np.linalg.norm(f[-1,:]))
            print('Boundary error, {}: {}'.format(field_names[field_index], error))

    fig.suptitle('Eigenvalue: {:1.4e}'.format(evalue))
    fig.show()


def plot_solution(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, plot_evalues, plot_fields):
    # Load the data
    filename = pickle_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    evalues, evectors = data['evalues'], data['evectors']

    if plot_fields:
        ns, neta = 256, 255
        s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
        if Lmax*Nmax < 2400:  # About 6GB in basis storage for (ns,neta) = (256,255)
            bases = create_bases(m, Lmax, Nmax, boundary_method, s, eta)
        else:
            bases = None
        domain = {'m': m, 'Lmax': Lmax, 'Nmax': Nmax, 's': s, 'eta': eta, 'boundary_method': boundary_method}
        onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, Lmax, Nmax, s, eta, bases, domain)
    else:
        onpick = None

    fig, ax = plot_spectrum(evalues, onpick)
    ax.set_title('Boussinesq Eigenvalues')
    ax.set_xlabel('Real(λ)')
    ax.set_ylabel('Imag(λ)')

    plot_filename_prefix = pickle_filename(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, directory='figures')[:-5]
    plot_filename = plot_filename_prefix + '.png'
    plt.savefig(plot_filename)

    fig.show()


def rotation_configs():
    return [{'Ekman': 10**-4,   'm': 6,  'omega': -.43346, 'Rayleigh': 5.1549, 'Lmax': 16, 'Nmax': 16},
            {'Ekman': 10**-4.5, 'm': 9,  'omega': -.44276, 'Rayleigh': 4.7613, 'Lmax': 16, 'Nmax': 16},
            {'Ekman': 10**-5,   'm': 14, 'omega': -.45715, 'Rayleigh': 4.5351, 'Lmax': 20, 'Nmax': 20},
            {'Ekman': 10**-5.5, 'm': 20, 'omega': -.45760, 'Rayleigh': 4.3937, 'Lmax': 28, 'Nmax': 40},
            {'Ekman': 10**-6,   'm': 30, 'omega': -.46394, 'Rayleigh': 4.3021, 'Lmax': 60, 'Nmax': 60},
            {'Ekman': 10**-6.5, 'm': 44, 'omega': -.46574, 'Rayleigh': 4.2416, 'Lmax': 16, 'Nmax': 32},
            {'Ekman': 10**-7,   'm': 65, 'omega': -.46803, 'Rayleigh': 4.2012, 'Lmax': 16, 'Nmax': 32},
            {'Ekman': 10**-7.5, 'm': 95, 'omega': -.46828, 'Rayleigh': 4.1742, 'Lmax': 30, 'Nmax': 120}]

def main():
    solve = True
    plot_spy = False
    plot_evalues = True
    plot_fields = True
    boundary_method = 'galerkin'
    nev = 3

    config_index = 7
    config = rotation_configs()[config_index]

    m, Ekman, Prandtl, Rayleigh, omega = config['m'], config['Ekman'], 1, config['Rayleigh'], config['omega']
    Lmax, Nmax = config['Lmax'], config['Nmax']

    omega = 1j*omega/Ekman**(2/3)
    Rayleigh /= Ekman**(1/3)

    print(f'Linear onset, m = {m}, Ekman = {Ekman:1.4e}, Prandtl = {Prandtl}, Rayleigh = {Rayleigh:1.4e}')
    print('  Domain size: Lmax = {}, Nmax = {}'.format(Lmax, Nmax))
    print('  Boundary method = ' + boundary_method)

    if solve:
        solve_eigenproblem(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, omega, plot_spy, nev=nev)

    if plot_fields or plot_evalues:
        plot_solution(m, Lmax, Nmax, boundary_method, Ekman, Prandtl, Rayleigh, plot_evalues, plot_fields)
        plt.show()


def analyze_evalues():
    config_index = 4
    config = rotation_configs()[config_index]

    m, Ekman, Prandtl, Rayleigh, omega = config['m'], config['Ekman'], 1, config['Rayleigh'], config['omega']
    boundary_method = 'galerkin'

    omega = 1j*omega/Ekman**(2/3)
    Rayleigh /= Ekman**(1/3)

    Llores, Nlores = 48, 48
    Lhires, Nhires = 52, 52

    def load_evalues(L, N):
        filename = pickle_filename(m, L, N, boundary_method, Ekman, Prandtl, Rayleigh)
        data = pickle.load(open(filename, 'rb'))
        return data['evalues'], data['evectors']
        
    evalues_lores, evectors_lores = load_evalues(Llores, Nlores)
    evalues_hires, _ = load_evalues(Lhires, Nhires)
    evalues, indices = discard_spurious_eigenvalues(evalues_lores, evalues_hires, cutoff=1e4, plot=True)
    evectors = evectors_lores[:,indices]

    print('Number of good eigenvalues: {}/{}'.format(len(evalues), len(evalues_lores)))

    if True:
        Lmax, Nmax = Llores, Nlores
        ns, neta = 256, 255
        s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
        bases = create_bases(m, Lmax, Nmax, boundary_method, s, eta)
        onpick = lambda index: plot_spectrum_callback(index, evalues, evectors, Lmax, Nmax, s, eta, bases)
    else:
        onpick = None

    fig, ax = plot_spectrum(evalues, onpick)
    ax.set_title('Resolved Boussinesq Eigenvalues')
    plt.show()


if __name__=='__main__':
    main()
#    analyze_evalues()

