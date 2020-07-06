import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from dedalus_sphere import jacobi as Jacobi
from dedalus_sphere import zernike as Zernike
from kronecker import Kronecker
from eigtools import eigsort
import os
import pickle


g_solve_eigenproblem = True
g_plot_fields = True
g_plot_evalues = True
g_plot_slices = True

# Jacobi operators
A = Jacobi.operator('A')
B = Jacobi.operator('B')
DZ = Jacobi.operator('D')
Z = Jacobi.operator('Z')
IdZ = Jacobi.operator('Id')

# Zernike operators
dim = 2
DR = Zernike.operator(dim, 'D')
E = Zernike.operator(dim, 'E')
R = Zernike.operator(dim, 'R')
IdR = Zernike.operator(dim, 'Id')

# Composite Jacobi operators
Z2 = Z**2                                     # Z**2
ZdZ = Z @ DZ(+1)                              # Z * d/dZ
AB = A(+1) @ B(+1)                            # a,b -> a+1,b+1
AB2 = AB**2                                   # a,b -> a+2,b+2

# Composite Zernike operators
E2 = E(+1)**2
R2 = R(-1) @ R(+1)                            # R**2
H = IdR - R2                                  # 1 - R**2
H2 = H**2                                     # (1 - R**2)**2


# Define custom lazily-evaluated zernike operators
def zernike_operator(dimension,name,radius=1):

    if name == 'RdR':
        def function(n,k,l):
            R, D = Zernike.operator(dimension,'R',radius=radius), Zernike.operator(dimension,'D',radius=radius)
            if l == 0:
                op = R(-1) @ D(+1)
            else:
                op = (R(+1) @ D(-1) + R(-1) @ D(+1))/2
            return op(n,k,l)
        return Jacobi.Operator(function,Zernike.ZernikeCodomain(0,1,0))

    return Zernike.operator(dimension,name,radius=radius)


RdR = zernike_operator(dim, 'RdR',radius=1)


def galerkin_matrix_2d(Nmax, k, m, Lmax, a, b):
    Sr = sparse.eye(Nmax)
    bc = Jacobi.polynomials(Lmax,a,b,1.0)
    Sz = sparse.diags([np.concatenate((bc[2:],[1,1])), -bc[:-2]], [0, -2])
    return sparse.kron(Sr, Sz)


def galerkin_matrix(Nmax, k, m, Lmax, a, b):
    Sp = galerkin_matrix_2d(Nmax, k,m+1, Lmax, a, b)
    Sm = galerkin_matrix_2d(Nmax, k,m-1, Lmax, a, b)
    S0 = galerkin_matrix_2d(Nmax, k,m,   Lmax, a, b)
    I = sparse.eye(Nmax*Lmax)
    return sparse.block_diag([Sp,Sm,S0,I,S0])


def make_square_resizer(dimensions):
    def resizer(dimension, mat):
        n = dimensions[dimension]
        mat = mat[:n, :n]
        ncols = np.shape(mat)[1]
        if ncols < n:
            mat = sparse.hstack([mat, np.zeros((n, n - ncols))])
        return mat
    return resizer


def laplacian(resizer):
    """Scaled laplacian operator
       L f = H**2 Lap f = H**2 Lap_H f + [2 H S d/ds eta d/deta
             + (1 - S**2*(1-eta**2))*(d/deta)**2 + (2 + S**2)eta d/deta] f
    """
    kron = lambda ops, opz: Kronecker([ops,opz], resizer=resizer)

    # opa: H**2 * Lap f
    opas = E(-1)**2 @ DR(-1) @ DR(+1)
    opaz = AB2

    # opb: 2 * H * (R * d/dR) * (Z * d/dZ)
    opbs = 2 * E(-1) @ RdR
    opbz = AB @ ZdZ

    # opc: (H + R**2 * Z**2) * (d/dZ)**2
    #   opc1: H * (d/dZ)**2
    #   opc2: R**2 * Z**2 * (d/dZ)**2
    opcs1 = H
    opcz1 = DZ(+1)**2
    opcs2 = R2
    opcz2 = ZdZ**2

    # opd: (2 + R**2) * (Z * d/dZ)
    opds = 2*IdR + R2
    opdz = AB @ ZdZ
   
    opa = kron(opas, opaz)
    opb = kron(opbs, opbz)
    opc = kron(opcs1, opcz1) + kron(opcs2, opcz2)
    opd = kron(opds, opdz)
    op = opa + opb + opc + opd

    return op


def laplacian_correction(resizer):
    """Laplacian correction for W = h*w"""
    kron = lambda ops, opz: Kronecker([ops,opz], resizer=resizer)

    # 2 + R**2 + 2 * H * (R * d/dR)
    ops1 = 2*IdR + R2 + 2 * E(-1) @ RdR
    opz1 = AB2

    # 2 * R**2 * (Z * d/dZ)
    ops2 = 2 * R2
    opz2 = AB @ ZdZ

    op1 = kron(ops1, opz1)
    op2 = kron(ops2, opz2)
    op = op1 + op2

    return op


def matrices(Nmax, k, m, Lmax, a, b, Ekman, Prandtl, Rayleigh, boundary_condition, boundary_method):
    """Construct matrices for X = [u+,u-,W,p,T]
    """
    resizer = make_square_resizer([Nmax, Lmax])
    kron = lambda ops, opz: Kronecker([ops, opz], resizer=resizer)

    Zero = 0.*kron(IdR, IdZ)
    L = laplacian(resizer)
    A = laplacian_correction(resizer)
    xi = np.sqrt(0.5)

    # Time derivative matrices
    M00 = Ekman * kron(H2, AB2)
    M11 = Ekman * kron(H2, AB2)
    M22 = Ekman * kron(H2, AB2)
    M33 = Zero
    M44 = Prandtl * kron(H2, AB2)

    # u+ equation - spin+ velocity component
    L00 = Ekman * L - 1j * kron(H2, AB2)
    L01 = Zero
    L02 = Zero
    L03 = -xi * (kron(E(-1) @ H @ DR(+1), AB2) + kron(R(+1) @ H, AB @ ZdZ))
    L04 =  xi * Rayleigh * kron(R(+1) @ H2, AB2)

    # u- equation - spin- velocity component
    L10 = Zero
    L11 = Ekman * L + 1j * kron(H2, AB2)
    L12 = Zero
    L13 = -xi * (kron(E(-1) @ H @ DR(-1), AB2) + kron(R(-1) @ H, AB @ ZdZ))
    L14 =  xi * Rayleigh * kron(R(-1) @ H2, AB2)

    # w equation - vertical velocity component
    L20 = Zero
    L21 = Zero
    L22 = Ekman * (L + A)
    L23 = -kron(H2, AB @ DZ(+1))
    L24 = Rayleigh * kron(H**3, AB2 @ Z)

    # Divergence equation
    L30 = xi * (kron(E(-1) @ DR(-1), AB) + kron(R(-1), ZdZ))
    L31 = xi * (kron(E(-1) @ DR(+1), AB) + kron(R(+1), ZdZ))
    L32 = kron(IdR, DZ(+1))
    L33 = Zero
    L34 = Zero

    # Temperature equation
    L40 = xi * Prandtl * kron(R(-1) @ H2, AB2)
    L41 = xi * Prandtl * kron(R(+1) @ H2, AB2)
    L42 = Prandtl * kron(H2, AB2 @ Z)
    L43 = Zero
    L44 = L

    # Construct the matrices
    uparams = [(Nmax,k,m+1),(Lmax,a,b)]  # u+
    vparams = [(Nmax,k,m-1),(Lmax,a,b)]  # u-
    wparams = [(Nmax,k,m),  (Lmax,a,b)]
    pparams = [(Nmax,k,m),  (Lmax,a,b)]
    Tparams = [(Nmax,k,m),  (Lmax,a,b)]
    param_sets = [uparams, vparams, wparams, pparams, Tparams]
    make_matrices = lambda mats: [mats[i](param_sets[i]) for i in range(5)]

    Mmats = make_matrices([M00, M11, M22, M33, M44])
    umats = make_matrices([L00, L01, L02, L03, L04])
    vmats = make_matrices([L10, L11, L12, L13, L14])
    wmats = make_matrices([L20, L21, L22, L23, L24])
    pmats = make_matrices([L30, L31, L32, L33, L34])
    Tmats = make_matrices([L40, L41, L42, L43, L44])

    verbose = True
    if verbose:
        print_codomains = lambda name, mats: [print('{}{}: {}'.format(name,i,mats[i].codomains)) for i in range(5)]
        print_codomains('M', [M00, M11, M22, M33, M44]); print()
        print_codomains('u', [L00, L01, L02, L03, L04]); print()
        print_codomains('v', [L10, L11, L12, L13, L14]); print()
        print_codomains('w', [L20, L21, L22, L23, L24]); print()
        print_codomains('p', [L30, L31, L32, L33, L34]); print()
        print_codomains('T', [L40, L41, L42, L43, L44])

    sparse_format = 'lil'
    M = sparse.block_diag(Mmats, format=sparse_format)
    L = sparse.bmat([umats, vmats, wmats, pmats, Tmats], format=sparse_format)

    if boundary_method == 'galerkin':
        S = galerkin_matrix(Nmax,k,m,Lmax,a,b)
        L = (L @ S).tolil()
        M = (M @ S).tolil()

        # Tau lines for last two coefficients - 2x2 identity
        bc_fields = [0,1,2,4]  # Fields with Galerkin basis
        nbc = 2                # Number of boundary conditions
        for field_index in bc_fields:
            offset = Nmax*Lmax*field_index
            for n in range(Nmax):
                index = offset + (n+1)*Lmax - nbc

                # Zero out the corresponding rows of M and L
                M[index:index+nbc,:] = 0
                L[index:index+nbc,:] = 0

                # Drop in the boundary conditions to the L matrix
                L[index:index+nbc,index:index+nbc] = sparse.eye(nbc)

    elif boundary_method == 'tau':
        eta_bc = np.array([-1., 1.])
        no_slip_mat = Jacobi.polynomials(Lmax,a,b,eta_bc,normalised=True).T

        if boundary_condition == 'no-slip':
            bcmat = no_slip_mat
        elif boundary_condition == 'stress-free':
            raise ValueError('stress-free boundary conditions not yet implemented')
        else:
            raise ValueError('Unsupported boundary condition type')

        bc_fields = [0,1,2,4]  # Fields with boundary conditions
        nbc = len(eta_bc)      # Number of boundary conditions
        for field_index in bc_fields:
            # Temperature field is always held at constant temperature
            if field_index == 4:
                bcmat = no_slip_mat

            offset = Nmax*Lmax*field_index
            for n in range(Nmax):
                row = offset + (n+1)*Lmax - nbc
                col = offset + n*Lmax

                # Zero out the corresponding rows of M and L
                M[row:row+nbc,:] = 0
                L[row:row+nbc,:] = 0

                # Drop in the boundary conditions to the L matrix
                L[row:row+nbc,col:col+Lmax] = bcmat

    else:
        raise ValueError('Unknown boundary method')

    M, L = M.tocsr(), L.tocsr()
    return M, L


def check_evectors(Nmax, Lmax, evalues, evectors):
    badvecs = []
    for i in range(len(evalues)):
        val, vec = evalues[i], evectors[:,i]
        if abs(val) < 1e-10:
            continue
        once = False
        for field in [0,1,2,4]:
            index = Nmax*Lmax*field + Lmax-2
            for n in range(Nmax):
                c = vec[index:index+2]
                if np.max(abs(c)) > 1e-15:
                    if not once:
                        print('eigenvector: {}, eigenvalue: {:1.3e}'.format(i, val))
                        once = True
                        badvecs.append(i)
                    print('  coef not zero: n = {}, coeffs = {}'.format(n, c))
                index += Lmax
    print('Number of bad eigenvectors: {}/{} (total: {})'.format(len(badvecs), len(evalues), 5*Nmax*Lmax))


def savedata(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def filename_prefix(directory='data'):
    basepath = os.path.join(os.path.dirname(__file__), directory)
    prefix = 'stretched_sphere'
    return os.path.join(basepath, os.path.join(prefix, prefix))


def pickle_filename(boundary_method, Nmax, Lmax, Ekman):
    return filename_prefix() + '-evalues-{}-Nmax={}-Lmax={}-Ekman={:1.3e}.pckl'.format(boundary_method,Nmax,Lmax,Ekman)


def solve_eigenproblem(config, Nmax, Lmax, boundary_method):
    k, a, b = 1/2, 0, 0  # Legendre polynomials since η doesn't appear in volume element dV = (1-s**2)**1/2 s ds dφ dη
    m = config['m']
    Ekman, Prandtl, Rayleigh = config['Ekman'], 1, config['Rayleigh']
    boundary_condition = 'no-slip'

    # Marti/Jones scaling
    Rayleigh /= Ekman**(1/3)

    # Construct the system
    M, L = matrices(Nmax, k, m, Lmax, a, b, Ekman, Prandtl, Rayleigh, boundary_condition, boundary_method)

    plot_spy = False
    if plot_spy:
        fig, plot_axes = plt.subplots(1,2,figsize=(9,4))
        plot_axes[0].spy(L)
        plot_axes[1].spy(M)

    # Compute the eigenvalues and eigenvectors
    print('Eigenvalue problem, size {}'.format(np.shape(L)))
    evalues, evectors = eigsort(L.todense(), M.todense(), profile=True)

    # Ensure the eigenvectors satisfy the boundary condition
    if boundary_method == 'galerkin':
        check_evectors(Nmax, Lmax, evalues, evectors)

    # Output data
    data = {'Nmax': Nmax, 'k': k, 'm': m, 
            'Lmax': Lmax, 'a': a, 'b': b,
            'Ekman': Ekman, 'Prandtl': Prandtl, 'Rayleigh': Rayleigh,
            'boundary_condition': boundary_condition,
            'boundary_method': boundary_method,
            'evalues': evalues, 'evectors': evectors}
    filename = pickle_filename(boundary_method, Nmax, Lmax, Ekman)
    savedata(filename, data)


def plotfield(s, eta, f):
    ss = np.reshape(s,(1,len(s)))
    ee = np.reshape(eta,(len(eta),1))
    zz = np.sqrt(1-ss**2)*ee

    fig, ax = plt.subplots(figsize=(4.25,6))
    im = ax.pcolormesh(ss, zz, f, cmap='RdBu')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('s')
    ax.set_ylabel('z')
    ax.set_aspect('equal', adjustable='box')
    fig.set_tight_layout(True)


def expansion(F, T, Q):
    return T @ (Q @ F).T


def plot_solution(config, Nmax, Lmax, boundary_method):
    # Load the data
    Ekman, omega = config['Ekman'], config['omega']
    filename = pickle_filename(boundary_method, Nmax, Lmax, Ekman)
    data = pickle.load(open(filename, 'rb'))

    # Extract configuration parameters
    k, m, a, b = data['k'], data['m'], data['a'], data['b']
    evalues, evectors = data['evalues'], data['evectors']
    plot_evalues, plot_slices, plot_fields = g_plot_evalues, g_plot_slices, g_plot_fields

    # Plot the eigenvalues
    if plot_evalues:
        plt.figure()
        plt.plot(evalues.real, evalues.imag, '.')
        plt.grid()
        plt.xlabel('Real(λ)')
        plt.ylabel('Imag(λ)')
        plt.title('Stretched Sphere Eigenvalues')

    # Get the target eigenpair
    target = 1j * omega * Ekman**(-2/3)
    index = np.argmin(abs(evalues - target))
    val, vec = evalues[index], evectors[:,index]

    # Recombine the Galerkin basis
    if boundary_method == 'galerkin':
        S = galerkin_matrix(Nmax,k,m,Lmax,a,b)
        vec = S @ vec

    # Extract coefficients 
    ncoeff = Nmax*Lmax
    upcoeff = vec[:ncoeff] 
    umcoeff = vec[ncoeff:2*ncoeff] 
    Wcoeff = vec[2*ncoeff:3*ncoeff]
    pcoeff = vec[3*ncoeff:4*ncoeff]
    Tcoeff = vec[4*ncoeff:5*ncoeff]

    # Construct the basis polynomials
    ns, neta = 1000, 1001
    s, eta = np.linspace(0,1,ns+1)[1:], np.linspace(-1,1,neta)
    z = 2*s**2 - 1
    Qn = Zernike.polynomials(dim,Nmax,k,m,z).T
    Qnp = Zernike.polynomials(dim,Nmax,k,m+1,z).T
    Qnm = Zernike.polynomials(dim,Nmax,k,m-1,z).T
    Tn = Jacobi.polynomials(Lmax,a,b,eta).T

    # Convert to grid space
    upgrid = expansion(np.reshape(upcoeff, (Nmax,Lmax)), Tn, Qnp)
    umgrid = expansion(np.reshape(umcoeff, (Nmax,Lmax)), Tn, Qnm)
    u =       np.sqrt(0.5) * (upgrid + umgrid)
    v = -1j * np.sqrt(0.5) * (upgrid - umgrid)
    W = expansion(np.reshape(Wcoeff, (Nmax,Lmax)), Tn, Qn)
    p = expansion(np.reshape(pcoeff, (Nmax,Lmax)), Tn, Qn)
    T = expansion(np.reshape(Tcoeff, (Nmax,Lmax)), Tn, Qn)

    scale = np.reshape(np.sqrt(1-s**2), (1,ns))
    scale[scale == 0.] = 1.
    w = W / scale

    print('Plotting eigenvector with eigenvalue {:1.4f}'.format(val))

    # Plot settings
    field_names = ['u','v','w','p','T']
    field_data = [u,v,w,p,T]
    plot_fields = [0,1,2,4]
    eta_indices = [neta//2, neta-1]

    if plot_slices:
        # Radial plot figure
        fig, radial_axes = plt.subplots(1, len(plot_fields))
        if len(plot_fields) == 1: radial_axes = [radial_axes]

        # Vertical plot figure
        fig2, vertical_axes = plt.subplots()

    # Plot the fields
    for i in range(len(plot_fields)):
        field = plot_fields[i]
        Fgrid = field_data[field]

        # Plot the field
        if plot_fields:
            print('Relative imaginary part: {:1.3f}'.format(np.sum(abs(Fgrid.imag)**2) / np.sum(abs(Fgrid)**2)))
            plotfield(s, eta, Fgrid.real)
            plt.title(field_names[field])

        # Plot radial slices
        if plot_slices:
            radial_axes[i].set_title(field_names[field])
            for j in range(len(eta_indices)):
                radial_part = Fgrid[eta_indices[j],:]
                etavalue = eta[eta_indices[j]].item()
                radial_axes[i].plot(s, radial_part.real, label='η = {}'.format(etavalue))

            # Plot vertical slices
            vertical_part = Fgrid[:,-1]
            vertical_axes.plot(eta, vertical_part.real, label=field_names[field])

    if plot_slices:
        # Add legends and grids to the radial plots
        for ax in radial_axes:
            ax.legend()
            ax.grid()

        # Clean up the vertical plots
        vertical_axes.set_title('Vertical slice at the equator')
        vertical_axes.set_xlabel('η')
        vertical_axes.legend()
        vertical_axes.grid()


def main():
    configs = [{'Ekman': 10**-4,   'm': 6,  'omega': -.43346, 'Rayleigh': 5.1549},
               {'Ekman': 10**-4.5, 'm': 9,  'omega': -.44276, 'Rayleigh': 4.7613},
               {'Ekman': 10**-5,   'm': 14, 'omega': -.45715, 'Rayleigh': 4.5351},
               {'Ekman': 10**-5.5, 'm': 20, 'omega': -.45760, 'Rayleigh': 4.3937},
               {'Ekman': 10**-6,   'm': 30, 'omega': -.46394, 'Rayleigh': 4.3021},
               {'Ekman': 10**-6.5, 'm': 44, 'omega': -.46574, 'Rayleigh': 4.2416},
               {'Ekman': 10**-7,   'm': 65, 'omega': -.46803, 'Rayleigh': 4.2012},
               {'Ekman': 10**-7.5, 'm': 95, 'omega': -.46828, 'Rayleigh': 4.1742}]

    config = configs[1]
    Nmax, Lmax, Ekman = 12, 12, config['Ekman']
    boundary_method = 'galerkin'

    print('Config: {}'.format(config))
    print('  Domain size: Nmax = {}, Lmax = {}'.format(Nmax, Lmax))
    print('  Boundary method: ' + boundary_method)

    if g_solve_eigenproblem:
        solve_eigenproblem(config, Nmax, Lmax, boundary_method)

    if g_plot_fields or g_plot_evalues or g_plot_slices:
        plot_solution(config, Nmax, Lmax, boundary_method)
        plt.show()


if __name__=='__main__':
    main()

