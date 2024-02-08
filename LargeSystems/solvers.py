import numpy as np
from LargeSystems.solution import Solution
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from copy import deepcopy
import warnings

def inner(x, y):
    # Numpy's inner product is broken, this fixes it
    return np.dot(x.T, y)[0,0]

def naturals():
    # A helper method to loop over the natural numbers, the cg algorithm is formatted differently
    k = 0
    while True:
        yield k
        k += 1

class SimpleSolver:
    # Solve Ax = b using numpy linalg solve in a dense format
    def __init__(self, sparse=False):
        # If sparse, convert matrices to CSR before calculations
        self.sparse = sparse

    def solve(self, system):
        # Sovle the build system, return a solution vector
        if self.sparse:
            x = spsolve(csr_matrix(system.A), system.b)
        else:
            x = np.linalg.solve(system.A, system.b)

        return Solution(x, system.var_idxs)
    
class GradientMethodSolver:
    # Solve Ax = b using the gradient method
    def __init__(self, sparse=False, x0=None, abs_tol=1e-7, maxit=100):
        # Initialize a gradient method solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # abs_tol is the absolute tolerance of the resulting residual
        # maxit is the maximum number of iterations
        self.sparse = sparse
        self.x0 = x0
        self.abs_tol = abs_tol
        self.maxit = maxit

    def solve(self, system):
        # Solve using the gradient method with exact step size
        # Return solution x, list of residuals, and last iteration
        # Print warnings if necessary
        A = system.A
        b = system.b
        x = deepcopy(self.x0)

        if self.sparse:
            A = csr_matrix(A)
        
        r = dict() # We use a dictionary for better visual indexing

        if x is None:
            # Shorthand for use the 0 vector
            x = np.zeros((A.shape[1], 1), dtype=np.float64)

        # The algorithm is followed exactly from page 183 of the notes
        r[0] = b - A * x
        k = 0

        # Stop if and only if:
        #   We found residual r so that r.T * A * r <= 0, so A is not SPD
        #   k >= maxiter
        #   norm(r) <= tol
        while (inner(r[k], A * r[k]) > 0) and (k < self.maxit) and (np.linalg.norm(r[k]) > self.abs_tol):
            a = np.linalg.norm(r[k]) ** 2 / inner(r[k], A * r[k])
            x += a * r[k]
            r[k + 1] = r[k] - a * A * r[k]
            k += 1
        
        if inner(r[k], A * r[k]) <= 0:
            # A is not SPD. 
            warnings.warn("Matrix A was not Symmetric Positive Definite")
            return x, list(r.values()), k
        
        elif k >= self.maxit:
            # We exceeded the number of iterations
            warnings.warn("The maximum number of iterations was exceeded")
            return x, list(r.values()), k
        
        # We are good!
        return x, list(r.values()), k
    
class ConjugateGradientMethodSolver:
    # Solve Ax = b using the conjugate gradient method
    def __init__(self, sparse=False, x0=None, abs_tol=1e-7, maxit=100):
        # Initialize a gradient method solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # abs_tol is the absolute tolerance of the resulting residual
        # maxit is the maximum number of iterations
        self.sparse = sparse
        self.x0 = x0
        self.abs_tol = abs_tol
        self.maxit = maxit

    def solve(self, system):
        # Solve using the gradient method with exact step size
        # Return solution x, list of residuals, and last iteration
        # Print warnings if necessary
        A = system.A
        b = system.b
        x = deepcopy(self.x0)

        if self.sparse:
            A = deepcopy(csr_matrix(A))
        
        r = dict() # We use a dictionary for easier visual indexing

        if x is None:
            # Shorthand for use the 0 vector
            x = np.zeros((A.shape[1], 1), dtype=np.float64)
        else:
            # Copy x so we dont modify the parameter
            x = deepcopy(x)

        # The algorithm is followed exactly from page 196 of the notes
        p = r[0] = b - A * x
        
        for k in naturals():
            if np.linalg.norm(r[k]) <= self.abs_tol:
                # We are below the tolerance, nice!
                return x, r, k
            elif inner(p, A * p) <= 0:
                # A is not positive definite :(
                warnings.warn("Matrix A was not Symmetric Positive Definite")
                return x, r, k
            elif k >= self.maxit:
                # Too many iterations :(
                warnings.warn("The maximum number of iterations was exceeded")
                return x, r, k
            
            a = inner(r[k], p) / inner(A * p, p)
            x += a * p
            r[k + 1] = r[k] - a * A * p
            b = np.linalg.norm(r[k + 1]) ** 2 / np.linalg.norm(r[k]) ** 2
            p = r[k + 1] + b * p