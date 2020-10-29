import numpy as np
import scipy.sparse as sparse
from dedalus_sphere import ball128
from dedalus_sphere import sphere128


def make_Q(ell_range,R_max):
    Q = {}
    for ell in ell_range:
        Q[(ell,0)] = np.array([[1]])
        for deg in range(1,R_max+1):
            Q[(ell,deg)] = ball128.recurseQ(Q[(ell,deg-1)],ell,deg)
    return Q


def BC_rows(N):
    N0 = N
    N1 = N + N0 + 1
    N2 = N + N1 + 1
    N3 = N + N2 + 1
    N4 = N + N3 + 1
    return N0,N1,N2,N3,N4


def matrices(B,N,ell,Ekman,Prandtl,Rayleigh,alpha_BC=0,implicit_buoyancy=True,implicit_base_temp=True,unscaled=False,boundary_condition='stress-free'):

    def op(op_name,N,k,ell,a=B.a,dtype=np.float64):
        return ball128.operator(op_name,N,k,ell,a=a).astype(dtype)

    def xi(mu,ell):
        # returns xi for ell > 0 or ell = 0 and mu = +1
        # otherwise returns 0.
        if (ell > 0) or (ell == 0 and mu == 1):
            return ball128.xi(mu,ell)
        return 0.

    def D(mu,i,deg):
        if mu == +1: return op('D+',N,i,ell+deg)
        if mu == -1: return op('D-',N,i,ell+deg)

    def R(mu,i,deg):
        """Multiplication by r"""
        if mu == +1: return op('R+',N,i,ell+deg)
        if mu == -1: return op('R-',N,i,ell+deg)

    def E(i,deg): return op('E',N,i,ell+deg)

    def C(deg): return ball128.connection(N,ell+deg,alpha_BC,2)

    Z = op('0',N,0,ell)

    N0, N1, N2, N3, N4 = BC_rows(N)

    if ell == 0:
        I = op('I',N,0,ell).tocsr()
        M44 = Prandtl*E(1, 0).dot(E( 0, 0))

        if implicit_base_temp:
            L42 = -Prandtl *  xi(+1, ell) * E(1, 0).dot(E(0, 0).dot(R(-1, 0, +1)))
        else:
            L42 = Z

        M = sparse.bmat([[Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,Z],
                         [Z,Z,Z,Z,M44]]).tocsr()
        L = sparse.bmat([[I,Z,Z,Z,Z],
                         [Z,I,Z,Z,Z],
                         [Z,Z,I,Z,Z],
                         [Z,Z,Z,I,Z],
                         [Z,Z,L42,Z,-D(-1,1,+1).dot(D(+1, 0, 0))]]).tocsr()

        row0=np.concatenate(( np.zeros(N3+1), op('r=1',N,0,ell) ))

        tau0 = C(0)[:,-1]
        tau0 = tau0.reshape((len(tau0),1))

        col0 = np.concatenate((np.zeros((N3+1,1)),tau0))

        L = sparse.bmat([[   L, col0],
                         [row0,    0]])

        M = sparse.bmat([[     M, 0*col0],
                         [0*row0,      0]])

        L = L.tocsr()
        M = M.tocsr()

        return M, L

    xim, xip = xi([-1,+1],ell)

    M00 = Ekman*E(1,-1).dot(E( 0,-1))
    M11 = Ekman*E(1, 0).dot(E( 0, 0))
    M22 = Ekman*E(1,+1).dot(E( 0,+1))
    M44 = Prandtl*E(1, 0).dot(E( 0, 0))

    M=sparse.bmat([[M00, Z,   Z,  Z,   Z],
                   [Z, M11,   Z,  Z,   Z],
                   [Z,   Z, M22,  Z,   Z],
                   [Z,   Z,   Z,  Z,   Z],
                   [Z,   Z,   Z,  Z, M44]])
    M = M.tocsr()

    L00 = -Ekman*D(-1,1, 0).dot(D(+1, 0,-1))
    L11 = -Ekman*D(-1,1,+1).dot(D(+1, 0, 0))
    L22 = -Ekman*D(+1,1, 0).dot(D(-1, 0,+1))
    L44 = -D(-1,1,+1).dot(D(+1, 0, 0))

    L03 = xim*E(+1,-1).dot(D(-1,0,0))
    L23 = xip*E(+1,+1).dot(D(+1,0,0))

    L30 = xim*D(+1,0,-1)
    L32 = xip*D(-1,0,+1)

    # Temperature-driven buoyancy forces
    if implicit_buoyancy:
        L04 = -Rayleigh * xim * E(1, -1).dot(E(0, -1).dot(R(-1, 0, 0)))
        L24 = -Rayleigh * xip * E(1, +1).dot(E(0, +1).dot(R(+1, 0, 0)))
    else:
        L04 = Z
        L24 = Z

    if implicit_base_temp:
        L40 = -Prandtl * xim * E(1, 0).dot(E(0, 0).dot(R(+1, 0, -1)))
        L42 = -Prandtl * xip * E(1, 0).dot(E(0, 0).dot(R(-1, 0, +1)))
    else:
        L40 = Z
        L42 = Z

    L=sparse.bmat([[L00,  Z,   Z, L03, L04],
                   [Z,  L11,   Z,   Z,   Z],
                   [Z,    Z, L22, L23, L24],
                   [L30,  Z, L32,   Z,   Z],
                   [L40,  Z, L42,   Z, L44]])
    L = L.tocsr()

    Qdict = make_Q([ell],2)
    Q1 = Qdict[(ell,1)]
    Q2 = Qdict[(ell,2)]

    if ell == 1: rDmm = 0.*op('r=1',N,1,ell)
    else: rDmm = xi(-1,ell-1)*op('r=1',N,1,ell-2)*D(-1,0,-1)
    rDpm = xi(+1,ell-1)*op('r=1',N,1,ell  )*D(+1,0,-1)
    rDm0 = xi(-1,ell  )*op('r=1',N,1,ell-1)*D(-1,0, 0)
    rDp0 = xi(+1,ell  )*op('r=1',N,1,ell+1)*D(+1,0, 0)
    rDmp = xi(-1,ell+1)*op('r=1',N,1,ell  )*D(-1,0,+1)
    rDpp = xi(+1,ell+1)*op('r=1',N,1,ell+2)*D(+1,0,+1)
    rD = np.array([rDmm, rDm0, rDmp, 0.*rDmm, 0.*rDm0, 0.*rDmp, rDpm, rDp0, rDpp])
    QSm = Q2[:,::3].dot(rD[::3])
    QS0 = Q2[:,1::3].dot(rD[1::3])
    QSp = Q2[:,2::3].dot(rD[2::3])
    u0m = op('r=1',N,0,ell-1)*Q1[1,0]
    u0p = op('r=1',N,0,ell+1)*Q1[1,2]

    if boundary_condition == 'stress-free':
        row0=np.concatenate(( QSm[1]+QSm[3], QS0[1]+QS0[3] , QSp[1]+QSp[3], np.zeros(N4-N2)))
        row1=np.concatenate(( u0m          , np.zeros(N0+1), u0p          , np.zeros(N4-N2)))
        row2=np.concatenate(( QSm[5]+QSm[7], QS0[5]+QS0[7] , QSp[5]+QSp[7], np.zeros(N4-N2)))
    elif boundary_condition == 'no-slip':
        row0 = np.concatenate((                op('r=1', N, 0, ell-1), np.zeros(N4-N0)))
        row1 = np.concatenate((np.zeros(N0+1), op('r=1', N, 0, ell),   np.zeros(N4-N1)))
        row2 = np.concatenate((np.zeros(N1+1), op('r=1', N, 0, ell+1), np.zeros(N4-N2)))
    else:
        raise ValueError('Unknown boundary condition')
    row3=np.concatenate(( np.zeros(N3+1), op('r=1',N,0,ell) ))

    tau0 = C(-1)[:,-1]
    tau1 = C( 0)[:,-1]
    tau2 = C( 1)[:,-1]
    tau3 = C( 0)[:,-1]

    tau0 = tau0.reshape((len(tau0),1))
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = tau2.reshape((len(tau2),1))
    tau3 = tau3.reshape((len(tau3),1))

    col0 = np.concatenate((                   tau0,np.zeros((N4-N0,1))))
    col1 = np.concatenate((np.zeros((N0+1,1)),tau1,np.zeros((N4-N1,1))))
    col2 = np.concatenate((np.zeros((N1+1,1)),tau2,np.zeros((N4-N2,1))))
    col3 = np.concatenate((np.zeros((N3+1,1)),tau3))

    L = sparse.bmat([[   L, col0, col1, col2, col3],
                     [row0,    0 ,   0,    0,    0],
                     [row1,    0 ,   0,    0,    0],
                     [row2,    0,    0,    0,    0],
                     [row3,    0,    0,    0,    0]])

    M = sparse.bmat([[     M, 0*col0, 0*col1, 0*col2, 0*col3],
                     [0*row0,      0,      0,      0,      0],
                     [0*row1,      0,      0,      0,      0],
                     [0*row2,      0,      0,      0,      0],
                     [0*row3,      0,      0,      0,      0]])

    L = L.tocsr()
    M = M.tocsr()

    return M, L


def rescale_system(ell, L, M, Ekman, Prandtl, Rayleigh, inplace=False):
    ntau = 1 if ell == 0 else 4
    N = (np.shape(L)[0]-ntau)//5

    if not inplace:
        L = sparse.csr_matrix.copy(L)
        M = sparse.csr_matrix.copy(M)

    # Ekman scalings
    M[   :  N,   :  N] *= Ekman     # du^-/dt
    M[  N:2*N,  N:2*N] *= Ekman     # du^0/dt
    M[2*N:3*N,2*N:3*N] *= Ekman     # du^+/dt

    L[   :  N,   :  N] *= Ekman     # Lap(u)^-
    L[  N:2*N,  N:2*N] *= Ekman     # Lap(u)^0
    L[2*N:3*N,2*N:3*N] *= Ekman     # Lap(u)^+

    # Prandtl scalings
    M[4*N:5*N,4*N:5*N] *= Prandtl   # dT/dt

    L[4*N:5*N,4*N:5*N] *= Prandtl   # Lap(T)
    L[4*N:5*N,   :  N] *= Prandtl   # (u.r)^-
    L[4*N:5*N,2*N:3*N] *= Prandtl   # (u.r)^+

    # Rayleigh scalings
    L[:N,     4*N:5*N] *= Rayleigh  # (T r)^-
    L[2*N:3*N,4*N:5*N] *= Rayleigh  # (T r)^+

    return L, M


def create_coriolis_matrix(B, state_vector, m_min=None, m_max=None):
    """Construct the Coriolis matrix acting on a vector field.  No tau lines are included"""
    L_max, N_max, R_max = B.L_max, B.N_max, B.R_max
    m_start = B.m_min if m_min is None else m_min
    m_end = B.m_max if m_max is None else m_max

    lrm2ind = state_vector.indexer('u')

    vmoff = lambda ell: 0 * state_vector.n_size[ell]
    v0off = lambda ell: 1 * state_vector.n_size[ell]
    vpoff = lambda ell: 2 * state_vector.n_size[ell]
    flatten = lambda z: [y for x in z for y in x]

    # row, col, values for each of the sparse matrices
    Qrows, Qcols, Qvals = [], [], []
    QTrows, QTcols, QTvals = [], [], []
    Mrows, Mcols, Mvals = [], [], []

    def op(handle, m, s):
        """Get an operator from the sphere and make it dense for ravel()"""
        return sphere128.operator(handle, L_max, m, s).todense().astype(np.float64)

    Qdict = make_Q(range(L_max+1),R_max=1)

    # Compute ez cross u in coefficient space
    for m in range(m_start, m_end+1):
        Cm  = -1j * op('C',  m, -1)
        Cp  =  1j * op('C',  m, +1)
        Spm = -1j * op('S+', m, -1) / 2**.5
        Smp =  1j * op('S-', m, +1) / 2**.5
        Sp0 =  1j * op('S+', m,  0) / 2**.5
        Sm0 = -1j * op('S-', m,  0) / 2**.5

        for n in range(0, N_max+1):
            ell_range = list(range(m, L_max+1))
            nn = len([ell for ell in ell_range if n < state_vector.n_size[ell]])

            for ell in ell_range:
                ell_local = ell-ell_range[0]
                if ell_local >= nn:
                    break
                Qell = Qdict[(ell,1)]
                QTell = Qell.T

                vmind = lrm2ind(ell, vmoff(ell) + n, m)
                v0ind = lrm2ind(ell, v0off(ell) + n, m)
                vpind = lrm2ind(ell, vpoff(ell) + n, m)

                Qrows += [vmind]*3 + [v0ind]*3 + [vpind]*3
                Qcols += [vmind, v0ind, vpind]*3
                Qvals += Qell.ravel().tolist()

                QTrows += [vmind]*3 + [v0ind]*3 + [vpind]*3
                QTcols += [vmind, v0ind, vpind]*3
                QTvals += QTell.ravel().tolist()

            if m == 0:
                e0rows = flatten([[lrm2ind(k+ell_range[0], v0off(k+ell_range[0]) + n, m)]*(nn-1) for k in range(nn)])
                e0cols = [lrm2ind(k+ell_range[0], v0off(k+ell_range[0]) + n, m) for k in range(nn)]*(nn-1)

                # Spin e(-) output
                Mrows += flatten([[lrm2ind(k+ell_range[0], vmoff(k+ell_range[0]) + n, m)]*(nn-1) for k in range(1,nn)])
                Mcols += [lrm2ind(k+ell_range[0], vmoff(k+ell_range[0]) + n, m) for k in range(1,nn)]*(nn-1)
                Mvals += np.ravel(Cm[:nn-1,:nn-1]).tolist()

                Mrows += flatten([[lrm2ind(k+ell_range[0], vmoff(k+ell_range[0]) + n, m)]*nn for k in range(1,nn)])
                Mcols += e0cols
                Mvals += np.ravel(Sm0[:nn-1,:nn]).tolist()

                # Spin e(0) output
                Mrows += e0rows
                Mcols += [lrm2ind(k+ell_range[0], vmoff(k+ell_range[0]) + n, m) for k in range(1,nn)]*nn
                Mvals += np.ravel(Spm[:nn,:nn-1]).tolist()

                Mrows += e0rows
                Mcols += [lrm2ind(k+ell_range[0], vpoff(k+ell_range[0]) + n, m) for k in range(1,nn)]*nn
                Mvals += np.ravel(Smp[:nn,:nn-1]).tolist()

                # Spin e(+) output
                Mrows += flatten([[lrm2ind(k+ell_range[0], vpoff(k+ell_range[0]) + n, m)]*nn for k in range(1,nn)])
                Mcols += e0cols
                Mvals += np.ravel(Sp0[:nn-1,:nn]).tolist()

                Mrows += flatten([[lrm2ind(k+ell_range[0], vpoff(k+ell_range[0]) + n, m)]*(nn-1) for k in range(1,nn)])
                Mcols += [lrm2ind(k+ell_range[0], vpoff(k+ell_range[0]) + n, m) for k in range(1,nn)]*(nn-1)
                Mvals += np.ravel(Cp[:nn-1,:nn-1]).tolist()

            else:
                vmrowinds = flatten([[lrm2ind(k+ell_range[0], vmoff(k+ell_range[0]) + n, m)]*nn for k in range(nn)])
                v0rowinds = flatten([[lrm2ind(k+ell_range[0], v0off(k+ell_range[0]) + n, m)]*nn for k in range(nn)])
                vprowinds = flatten([[lrm2ind(k+ell_range[0], vpoff(k+ell_range[0]) + n, m)]*nn for k in range(nn)])

                vmcolinds = [lrm2ind(k+ell_range[0], vmoff(k+ell_range[0]) + n, m) for k in range(nn)]*nn
                v0colinds = [lrm2ind(k+ell_range[0], v0off(k+ell_range[0]) + n, m) for k in range(nn)]*nn
                vpcolinds = [lrm2ind(k+ell_range[0], vpoff(k+ell_range[0]) + n, m) for k in range(nn)]*nn

                # Spin e(-) output
                Mrows += vmrowinds
                Mcols += vmcolinds
                Mvals += np.ravel(Cm[:nn,:nn]).tolist()

                Mrows += vmrowinds
                Mcols += v0colinds
                Mvals += np.ravel(Sm0[:nn,:nn]).tolist()

                # Spin e(0) output
                Mrows += v0rowinds
                Mcols += vmcolinds
                Mvals += np.ravel(Spm[:nn,:nn]).tolist()

                Mrows += v0rowinds
                Mcols += vpcolinds
                Mvals += np.ravel(Smp[:nn,:nn]).tolist()

                # Spin e(+) output
                Mrows += vprowinds
                Mcols += v0colinds
                Mvals += np.ravel(Sp0[:nn,:nn]).tolist()

                Mrows += vprowinds
                Mcols += vpcolinds
                Mvals += np.ravel(Cp[:nn,:nn]).tolist()

    # Throw away zeros
    inds = np.nonzero(Mvals)[0]
    Mrows, Mcols, Mvals = np.asarray(Mrows)[inds], np.asarray(Mcols)[inds], np.asarray(Mvals)[inds]

    # Construct the sparse matrix
    dof = state_vector.dof
    Q  = sparse.csr_matrix((Qvals,  (Qrows, Qcols)),   shape=(dof,dof))
    QT = sparse.csr_matrix((QTvals, (QTrows, QTcols)), shape=(dof,dof))
    M  = sparse.csr_matrix((Mvals,  (Mrows, Mcols)),   shape=(dof,dof))
    A = QT @ (M @ Q)

    return A
