import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix

class MatrixWithBoundaryConditions:
    # Store larger matrix with the boundary conditions as described in problem 1
    def __init__(self, n):
        # Define our matrix
        # We store the dimension and a scalar which left-multiplies it
        # In essence, this object is a*A. 
        self.n = n
        self.scalar = 1

    def __mul__(self, vec):
        # Right multiply out another vector
        ret = np.zeros((self.n + 2, 1), dtype=np.float64)

        ret[0] = self.scalar * vec[0] # first row
        ret[-1] = self.scalar * vec[-1] # last row

        for i in range(1, len(vec) - 1):
            # All intermediate rows are -1 2 -1
            ret[i] = -self.scalar * vec[i - 1] + 2 * self.scalar * vec[i] - self.scalar * vec[i + 1]

        return ret

    def __rmul__(self, val):
        # Left multiplication by float
        return MatrixWithBoundaryConditions(self.n, val * self.scalar)
    
    def aslinop(self):
        # Return object as a linear operator
        return LinearOperator((self.n + 2, self.n + 2), lambda vec: self * vec)
    
class Tridiagonal:
    # Simple class to accelerate multiplication of tridiagonal matrices
    def __init__(self, n, lower, diag, upper):
        # Create an nxn tridiagonal matrix with diag in the diagonal, and lower and upper in the lower and upper diagonal

        self.n = n
        self.shape = [n, n]
        self.lower = lower
        self.diag = diag
        self.upper = upper

    def __mul__(self, vec):
        # RIght multiply out another vector
        ret = np.zeros((self.n, 1), dtype=np.float64)

        # Top and bottom rows
        ret[0] = self.diag * vec[0] + self.upper * vec[1]
        ret[-1] = self.lower * vec[-2] + self.diag * vec[-1]

        # All intermediate rows
        for i in range(1, len(vec) - 1):
            ret[i] = self.lower * vec[i - 1] + self.diag * vec[i] + self.upper * vec[i + 1]

        return ret

    def __rmul__(self, val):
        # Left multiplication by float
        return Tridiagonal(self.n, val * self.lower, val * self.diag, val * self.upper)
    
    def aslinop(self):
        # Return object as a linear operator
        return LinearOperator((self.n, self.n), lambda vec: self * vec)
    
def compute_K(α, β, h, n):
    # Compute the matrix K for problem 4
    K = 1 / h**2 * (np.diag(np.full(n - 1, α + β*h), k=-1) 
                    + np.diag(np.full(n, -2*α - β*h), k=0) 
                    + np.diag(np.full(n - 1, α), k=1))
    
    K[0,n-1] = 1 / h**2 * (α + β*h)
    K[n-1,0] = 1 / h**2 * α

    return csr_matrix(K)

def compute_H(n, m):
    # Compute the matrix H for problem 4
    H = np.asmatrix(np.zeros((m, n), dtype=np.float64))
    for i in range(1, m + 1):
        H[i - 1, (n//m)*i - 1] = 1
    
    return csr_matrix(H)