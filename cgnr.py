import numpy as np
from enum import Enum
from copy import deepcopy
import warnings

class Tolerance(Enum):
    # Some tolerances for CGNR
    RELATIVE_RESIDUAL = 0
    RESIDUAL = 1
    ZERO = 2 # Zero tolerance, go for maximum iterations

def naturals():
    # A helper method to loop over the natural numbers, the cg algorithm is formatted differently
    k = 0
    while True:
        yield k
        k += 1

class CGNRSolver:
    # Solve least squares problem using CGNR
    def __init__(self, x0=None, tol_type=Tolerance.RESIDUAL, tol=1e-7, maxit=100):
        # Initialize a conjugate gradient method solver
        # x0 is the default starting vector, defaults to 0
        # Given tolerance type
        # tol is the tolerance level given tolerance type
        # maxit is the maximum number of iterations
        self.x0 = x0
        self.tol_type = tol_type
        self.tol = tol
        self.maxit = maxit

    def solve(self, A, At, b, n, metadata=None):
        # Solve least squares problem given functions for A, A transpose, and vector b
        # Must be given size n since functions A and At do not report it
        # Return solution x
        # Return metadata through optional metadata argument
        x = deepcopy(self.x0)

        if x is None:
            # Shorthand for use the 0 vector
            x = np.zeros((n, 1), dtype=np.float64)

        # The algorithm is followed exactly from page 203 of the notes
        r = b - A(x)
        Atr = At(r)
        p = Atr
        
        for k in naturals():
            if metadata is not None:
                metadata.save(x=x, r=r, p=p)

            if self.tol_type == Tolerance.RESIDUAL and np.linalg.norm(Atr) <= self.tol:
                # We are below the tolerance, nice!
                return x
            elif self.tol_type == Tolerance.RELATIVE_RESIDUAL and np.linalg.norm(Atr) / np.linalg.norm(At(b)) <= self.tol:
                # We are below the tolerance, nice!
                return x
            elif k >= self.maxit:
                # Too many iterations :(
                warnings.warn("The maximum number of iterations was exceeded")
                return x
            
            Ap = A(p)
            
            a = np.linalg.norm(Atr) ** 2 / np.linalg.norm(Ap) ** 2
            x += a * p

            new_r = r - a * Ap
            new_Atr = At(new_r)

            b = np.linalg.norm(new_Atr) ** 2 / np.linalg.norm(Atr) ** 2
            r = new_r
            Atr = new_Atr

            p = Atr + b * p