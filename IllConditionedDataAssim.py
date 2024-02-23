from copy import deepcopy
import numpy as np
from cgnr import CGNRSolver
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def solve(θ, n, m, Δt, nt, mt, W, K, H, z):
    '''
    Run the ill-conditioned data assim. problem outlined in 2.114
    Given parameters required
    Return the solution along with residual norms
    '''
    # We will need to solve the system 2.111 many times, so precompute the matrices
    matrix_2_111_left = np.asmatrix(np.identity(n) - (1 - θ) * Δt * K)
    matrix_2_111_right = np.asmatrix(np.identity(n) + θ * Δt * K)
    matrix_2_111_left = csr_matrix(matrix_2_111_left)
    matrix_2_111_right = csr_matrix(matrix_2_111_right)

    def A(v):
        # First define how to multiply matrix A
        # Algorithm outlined on page 86 to 87

        # Save all of the computed ws to be combined later
        ws = []

        y = deepcopy(v)
        for _ in range(mt):
            for _ in range(nt // mt):
                # Solve the system as outlined in the algorithm
                # Scipy likes making vectors horizontal, so we need to override this
                y = np.asmatrix(spsolve(matrix_2_111_left, matrix_2_111_right * y)).T
            
            ws.append(H * y)

        # Scale by weight matrix
        return W * np.vstack(ws)
    
    def At(w):
        # Define how to multiply matrix At
        # Algorithm outlined on page 87

        v = np.zeros((n, 1), dtype=np.float64)

        # First, scale by weight matrix
        w = W.T * w

        for j in range(mt - 1, -1, -1): # We include -1 to get down to 0
            v = v + H.T * w[m * j:m * j + m]
            for _ in range(nt // mt - 1, -1, -1):
                # Scipy likes making vectors horizontal, so we need to override this
                v = np.asmatrix(spsolve(matrix_2_111_left.T, v)).T
                v = matrix_2_111_right.T * v
        
        return v
    
    # Before solving, create a metadata object to store our residual norms
    class Metadata:
        def __init__(self):
            self.res_norms = []
        
        def save(self, r, **kwargs):
            self.res_norms.append(np.linalg.norm(At(r)))
    
    metadata = Metadata()
    
    # Now that we have our functions, we can solve!
    # Keep going until tolerance is less than 1e-5
    solver = CGNRSolver(maxit=float('inf'), tol=1e-5)

    # Return the solution and the residual norms to plot
    return solver.solve(A, At, z, n, metadata), metadata.res_norms