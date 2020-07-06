import numpy as np
import scipy.sparse as sparse

class Kronecker():
    def __init__(self, operators, resizer=None, is_leaf=True, ndim=None, codomains=None, Output=None):
        if is_leaf:
            self._operators = operators
            self._function = None
        else:
            self._operators = None
            self._function = operators

        if resizer is None:
            resizer = lambda dim, mat: mat
        self._resizer = resizer

        if ndim is None:
            ndim = len(operators)
        self._ndim = ndim

        if codomains is None:
            codomains = [op.codomain for op in self.operators]
        self._codomains = codomains

        if Output is None:
            Output = Kronecker
        self._Output = Output

      
    @property
    def operators(self):
        if not self.is_leaf:
            raise ValueError('Operators only valid for leaf node Kronecker product')
        return self._operators

    @property
    def function(self):
        if self.is_leaf:
            raise ValueError('Function only valid for non-leaf node Kronecker product')
        return self._function

    @property
    def ndim(self):
        return self._ndim

    @property
    def is_leaf(self):
        return self._operators is not None

    @property
    def Output(self):
        return self._Output

    @property
    def codomains(self):
        return self._codomains

    @property
    def resizer(self):
        return self._resizer

    def __call__(self, params):
        if len(params) != self.ndim:
            raise ValueError('Dimension mismatch!')

        if self.is_leaf:
            ndim = self.ndim
            # Compute the matrix operators
            mats = [self.operators[i](*params[i]) for i in range(ndim)]

            # Resize the operators
            mats = [self.resizer(i, mats[i]) for i in range(ndim)]

            # Finally kronecker out the matrices
            result = mats[0]
            for i in range(1, ndim):
                result = sparse.kron(result, mats[i])
            return result
        else:
            return self.function(params)

    def _check_dimension(self, other):
        if self.ndim != other.ndim:
            raise ValueError('Dimension mismatch!')

    def __matmul__(self, other):
        if not isinstance(other, Kronecker):
            raise ValueError('Matrix multiply only supported for two Kronecker instances')        
        self._check_dimension(other)
        ndim = self.ndim
        codomains = [self.codomains[index] + other.codomains[index] for index in range(ndim)]
        def function(*args):
            nextargs = [other.codomains[i](*args[0][i]) for i in range(self.ndim)]
            return self(nextargs) @ other(*args)
        return self.Output(function, is_leaf=False, ndim=ndim, codomains=codomains)

    def __add__(self, other):
        if not isinstance(other, Kronecker):
            raise ValueError('Can only add Kronecker instances together')
        self._check_dimension(other)
        ndim = self.ndim
        codomains = [self.codomains[index] | other.codomains[index] for index in range(ndim)]
        def function(*args):
            return self(*args) + other(*args)
        return self.Output(function, is_leaf=False, ndim=ndim, codomains=codomains)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not np.isscalar(other):
            raise ValueError('Only scalar product with Kronecker supported')
        def function(*args):
            return other * self(*args)
        return self.Output(function, is_leaf=False, ndim=self.ndim, codomains=self.codomains)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1/other)

    def __pos__(self):
        return self
    
    def __neg__(self):
        return (-1)*self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other


