import numpy as np

class StateVector:
    """Utility class to pack and unpack state vectors with specified layouts.
        This enables the user to specify ordering of data for an arbitrary set of tensor fields.
        The class is used to help index into matrix operators of specified layouts.  For example
        if problems are decoupled in both ell and m it makes sense to use the layout 'lmr' since
        it is block diagonal along this axis.  Coriolis is decoupled in m so eigenproblems are
        best performed in the 'mlr' layout.
    """
    def __init__(self, B, layout, fields, ntau=None, truncate=True, m_min=None, m_max=None):
        """Initialize the state vector with Ball B, layout and vector fields
           :param B: Dedalus Ball object
           :param layout: layout string specifying ordering of data in the state vector.
                Can be one of 'lrm', 'lmr' or 'mlr'
           :param fields: list of field names and ranks
           :param ntau: function handle that returns number of boundary conditions for a given ell
           :param truncate: indicate whether or not to discard invalid coefficients (m > ell)
        """
        if np.ndim(fields) == 1:
            fields = [fields]
        self.field_names, self.field_ranks = zip(*fields)
        if ntau is None:
            ntau = lambda _: 0

        L_max, N_max, R_max = B.L_max, B.N_max, B.R_max
        m_start = B.m_min if m_min is None else m_min
        m_end = B.m_max if m_max is None else m_max
        ell_start = m_start if truncate else 0
        ell_range = range(ell_start, L_max+1)

        # Number of m coefficients for a given ell
        if truncate:
            m_size = [min(ell+1, m_end-m_start+1) for ell in range(L_max+1)]
        else:
            m_size = [m_end - m_start + 1]*(L_max+1)

        # Number of radial coefficients for a given ell per regularity component
        # Vectors have 3*n_size[ell] radial coefficients for each m <= ell <= L_max
        n_size = [N_max+1 - max(ell - R_max, 0)//2 for ell in range(L_max+1)]

        # Shapes of each field for each ell, given as lshapes[field][ell] = (N_r, N_m)
        lshapes, lsums = {}, {}
        for name in self.field_names:
            lshapes[name], lsums[name] = [], []
        for name, rank in fields:
            lshapes[name] = [(3**rank*n_size[ell], m_size[ell]) for ell in ell_range]
        lshapes['tau'] = [(ntau(ell), m_size[ell]) for ell in ell_range]

        loffsets = {}
        cumsum = 0
        for name, rank in fields:
            loffsets[name] = []
        loffsets['tau'] = []
        for ell in ell_range:
            ell_local = ell-ell_start
            for name, rank in fields:
                loffsets[name].append(cumsum)
                cumsum += np.prod(lshapes[name][ell_local])
            loffsets['tau'].append(cumsum)
            cumsum += np.prod(lshapes['tau'][ell_local])
        dof = cumsum

        # Shapes of state vector for each m
        mshapes, moffsets = {}, {}
        for name in self.field_names:
            mshapes[name], moffsets[name] = [], []
        mshapes['tau'], moffsets['tau'] = [], []
        cumsum = 0
        for m in range(m_start, m_end+1):
            ell_start = m if truncate else 0
            for name, rank in fields:
                mshapes[name].append([3**rank*n_size[ell] for ell in range(ell_start, L_max+1)])
            mshapes['tau'].append([ntau(ell) for ell in range(ell_start, L_max+1)])

            for name in self.field_names:
                moffsets[name].append([])
            moffsets['tau'].append([])
            for ell in range(ell_start, L_max+1):
                for name, rank in fields:
                    moffsets[name][m-m_start].append(cumsum)
                    cumsum += mshapes[name][m-m_start][ell-ell_start]
                moffsets['tau'][m-m_start].append(cumsum)
                cumsum += mshapes['tau'][m-m_start][ell-ell_start]

        self.L_max = L_max
        self.truncate = truncate
        self.n_size = n_size
        self.m_size = m_size
        self.lshapes = lshapes
        self.loffsets = loffsets
        self.moffsets = moffsets
        self.dof = dof
        self.ell_start = m_start if truncate else 0
        self.m_start = m_start
        self.m_end = m_end

        # Finally set the layout
        self.layout = layout

    def index(self, name, l, r, m):
        return self._indexers[name](l, r, m)

    def indexer(self, name):
        return self._indexers[name]

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, s):
        supported_layouts = ['lrm', 'lmr', 'mlr']
        if s not in supported_layouts:
            raise ValueError("Unsupported layout - must be one of {}".format(supported_layouts))

        self._layout = s
        ell_start = self.ell_start
        m_start = self.m_start
        truncate = self.truncate

        if truncate:
            offset = [m for m in range(self.L_max + 1)]
        else:
            offset = [0] * (self.L_max + 1)

        def check(a):
            if a < 0:
                raise ValueError('Index is negative!')
            return a

        def make_indexer(name):
            lshapes = self.lshapes[name]
            loffsets = self.loffsets[name]
            moffsets = self.moffsets[name]
            if s == 'lrm':
                return lambda l, r, m: loffsets[check(l-ell_start)] + r * lshapes[check(l-ell_start)][1] + check(m-m_start)
            elif s == 'lmr':
                return lambda l, r, m: loffsets[check(l-ell_start)] + r + lshapes[check(l-ell_start)][0] * check(m-m_start)
            elif s == 'mlr':
                return lambda l, r, m: moffsets[check(m-m_start)][check(l-offset[m])] + check(r)

        indexers = {}
        for name in self.field_names:
            indexers[name] = make_indexer(name)
        indexers['tau'] = make_indexer('tau')
        self._indexers = indexers

    def pack(self, fields, output=None):
        if len(fields) != len(self.field_names):
            raise ValueError('Incorrect number of fields to pack!')

        if output is None:
            v = np.zeros(self.dof, dtype=np.complex128)
        else:
            if len(output != self.dof):
                raise ValueError('Incorrect output size')
            v = output

        for i in range(len(self.field_names)):
            name = self.field_names[i]
            rank = self.field_ranks[i]
            field = fields[i]
            if field.rank != rank:
                raise ValueError('Incorrect rank of field.  Are the fields sorted correctly?')

            indexer = self.indexer(name)
            ell_start = self.ell_start
            for ell in range(ell_start, self.L_max+1):
                m_size = ell+1-ell_start if self.truncate else self.L_max+1
                m_size = min(m_size, self.m_end+1-self.m_start)
                m_range = range(self.m_start, self.m_start+m_size)
                n_size = 3**rank*self.n_size[ell]
                inds = [indexer(ell, r, m) for r in range(n_size) for m in m_range]
                v[inds] = field['c'][ell][:n_size, m_range].ravel()
        return v

    def unpack(self, v, fields):
        if len(fields) != len(self.field_names):
            raise ValueError('Incorrect number of fields to pack!')
        for i in range(len(self.field_names)):
            name = self.field_names[i]
            rank = self.field_ranks[i]
            field = fields[i]
            if field.rank != rank:
                raise ValueError('Incorrect rank of field.  Are the fields sorted correctly?')
            field.layout = 'c'

            indexer = self.indexer(name)
            ell_start = self.ell_start
            for ell in range(ell_start, self.L_max+1):
                m_size = ell+1-ell_start if self.truncate else self.L_max+1
                m_size = min(m_size, self.m_end+1-self.m_start)
                m_range = range(self.m_start, self.m_start+m_size)
                n_size = 3**rank*self.n_size[ell]
                inds = [indexer(ell, r, m) for r in range(n_size) for m in m_range]
                field['c'][ell][:n_size, m_range] = np.reshape(v[inds], (n_size, m_size))

