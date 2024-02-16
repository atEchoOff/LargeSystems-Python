import numpy as np
from LargeSystems.solution import Solution
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, triu, tril, diags
from copy import deepcopy
import warnings
from enum import Enum

# Stopping conditions to be passed in
class Tolerance(Enum):
    RELATIVE_RESIDUAL = 0
    RESIDUAL = 1
    ZERO = 2 # Zero tolerance, go for maximum iterations

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
    # Solve Ax = b using numpy linalg solve
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
    def __init__(self, sparse=False, x0=None, tol_type=Tolerance.RESIDUAL, tol=1e-7, maxit=100):
        # Initialize a gradient method solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # Given tolerance type
        # tol is the tolerance level given tolerance type
        # maxit is the maximum number of iterations
        self.sparse = sparse
        self.x0 = x0
        self.tol_type = tol_type
        self.tol = tol
        self.maxit = maxit

    def solve(self, system, metadata=None):
        # Solve using the gradient method with exact step size
        # Return solution x
        # Return metadata through optional metadata argument
        A = system.A
        b = system.b
        x = deepcopy(self.x0)

        if self.sparse:
            A = csr_matrix(A)

        if x is None:
            # Shorthand for use the 0 vector
            x = np.zeros((A.shape[1], 1), dtype=np.float64)

        # The algorithm is followed exactly from page 183 of the notes
        r = b - A * x
        k = 0

        # Stop if and only if:
        #   We found residual r so that r.T * A * r <= 0, so A is not SPD
        #   k >= maxiter
        #   Matched tolerance
        while (inner(r, A * r) > 0) and (k < self.maxit):
            if metadata is not None:
                metadata.save(x=x, r=r)

            if self.tol_type == Tolerance.RESIDUAL and (np.linalg.norm(r) <= self.tol):
                break
            elif self.tol_type == Tolerance.RELATIVE_RESIDUAL and (np.linalg.norm(r) / np.linalg.norm(b) <= self.tol):
                break

            a = np.linalg.norm(r) ** 2 / inner(r, A * r)
            x += a * r
            r -= a * A * r
            k += 1
        
        if not all(r == 0) and inner(r, A * r) <= 0:
            # A is not SPD. 
            warnings.warn("Matrix A was not Symmetric Positive Definite")
            return Solution(x, system.var_idxs)
        
        elif k >= self.maxit:
            # We exceeded the number of iterations
            warnings.warn("The maximum number of iterations was exceeded")
            return Solution(x, system.var_idxs)
        
        # We are good!
        return Solution(x, system.var_idxs)
    
class ConjugateGradientMethodSolver:
    # Solve Ax = b using the conjugate gradient method
    def __init__(self, sparse=False, x0=None, tol_type=Tolerance.RESIDUAL, tol=1e-7, maxit=100):
        # Initialize a conjugate gradient method solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # Given tolerance type
        # tol is the tolerance level given tolerance type
        # maxit is the maximum number of iterations
        self.sparse = sparse
        self.x0 = x0
        self.tol_type = tol_type
        self.tol = tol
        self.maxit = maxit

    def solve(self, system, metadata=None):
        # Solve using the conjugate gradient method
        # Return solution x
        # Return metadata through optional metadata argument
        A = system.A
        b = system.b
        x = deepcopy(self.x0)

        if self.sparse:
            A = csr_matrix(A)

        if x is None:
            # Shorthand for use the 0 vector
            x = np.zeros((A.shape[1], 1), dtype=np.float64)

        # The algorithm is followed exactly from page 196 of the notes
        p = r = b - A * x
        
        for k in naturals():
            if metadata is not None:
                metadata.save(x=x, r=r, p=p)

            if self.tol_type == Tolerance.RESIDUAL and np.linalg.norm(r) <= self.tol:
                # We are below the tolerance, nice!
                return Solution(x, system.var_idxs)
            elif self.tol_type == Tolerance.RELATIVE_RESIDUAL and np.linalg.norm(r) / np.linalg.norm(b) <= self.tol:
                # We are below the tolerance, nice!
                return Solution(x, system.var_idxs)
            elif inner(p, A * p) <= 0:
                # A is not positive definite :(
                warnings.warn("Matrix A was not Symmetric Positive Definite")
                return Solution(x, system.var_idxs)
            elif k >= self.maxit:
                # Too many iterations :(
                warnings.warn("The maximum number of iterations was exceeded")
                return Solution(x, system.var_idxs)
            
            a = inner(r, p) / inner(A * p, p)
            x += a * p

            new_r = r - a * A * p
            b = np.linalg.norm(new_r) ** 2 / np.linalg.norm(r) ** 2
            r = new_r

            p = r + b * p

class JacobiMethodSolver:
    def __init__(self, sparse=False, x0=None, tol_type=Tolerance.RESIDUAL, tol=1e-7, maxit=100):
        # Initialize a jacobi method solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # Given tolerance type
        # tol is the tolerance level given tolerance type
        # maxit is the maximum number of iterations
        self.sparse = sparse
        self.x0 = x0
        self.tol_type = tol_type
        self.tol = tol
        self.maxit = maxit

    def solve(self, system, metadata=False):
        # Solve using the jacobi method
        # Return solution x and pass in metadata to optional metadata argument
        A = system.A
        b = system.b
        x = self.x0

        if self.sparse:
            A = csr_matrix(A)
            D = diags(A.diagonal(), format="csr") # diagonal component of A
            solve = spsolve
        else:
            D = np.diag(np.diag(A)) # diagonal component of A
            solve = np.linalg.solve
        DmA = D - A # Precalculate this so we dont need it later

        if x is None:
            x = np.zeros((A.shape[1], 1), dtype=np.float64)

        # Begin iteration
        for k in naturals():
            r = b - A * x
            if metadata is not None:
                metadata.save(x=x, r=r)

            if self.tol_type == Tolerance.RESIDUAL and np.linalg.norm(r) <= self.tol:
                # We are below the tolerance, nice!
                return Solution(x, system.var_idxs)
            elif self.tol_type == Tolerance.RELATIVE_RESIDUAL and np.linalg.norm(r) / np.linalg.norm(b) <= self.tol:
                # We are below the tolerance, nice!
                return Solution(x, system.var_idxs)
            elif k >= self.maxit:
                # Too many iterations :(
                warnings.warn("The maximum number of iterations was exceeded")
                return Solution(x, system.var_idxs)
            
            x = np.asmatrix(solve(D, DmA * x + b)) # D is diagonal, so this is fast
            if self.sparse:
                x = x.T # Sparse library transposes the solution... wth scipy

class GaussSeidelSolver:
    def __init__(self, sparse=False, x0=None, tol_type=Tolerance.RESIDUAL, tol=1e-7, maxit=100):
        # Initialize a gauss seidel solver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # Given tolerance type
        # tol is the tolerance level given tolerance type
        # maxit is the maximum number of iterations
        self.sparse = sparse
        self.x0 = x0
        self.tol_type = tol_type
        self.tol = tol
        self.maxit = maxit

    def solve(self, system, metadata=False):
        # Solve using Gauss-Seidel
        # Return solution x and pass in metadata to metadata object (optional)
        A = system.A
        b = system.b
        x = self.x0

        if self.sparse:
            A = csr_matrix(A)
            D = diags(A.diagonal(), format="csr") # diagonal component of A
            mU = D - triu(A, format="csr") # Negative strictly upper triangular component
            DpL = tril(A, format="csr") # D + L
            solve = spsolve
        else:
            D = np.diag(np.diag(A)) # diagonal component of A
            mU = np.asmatrix(D - np.triu(A)) # Negative strictly upper triangular component... really numpy?
            DpL = np.asmatrix(np.tril(A)) # D + L
            solve = np.linalg.solve

        if x is None:
            x = np.zeros((A.shape[1], 1), dtype=np.float64)

        # Begin iteration
        for k in naturals():
            r = b - A * x
            if metadata is not None:
                metadata.save(x=x, r=r)

            if self.tol_type == Tolerance.RESIDUAL and np.linalg.norm(r) <= self.tol:
                # We are below the tolerance, nice!
                return Solution(x, system.var_idxs)
            elif self.tol_type == Tolerance.RELATIVE_RESIDUAL and np.linalg.norm(r) / np.linalg.norm(b) <= self.tol:
                # We are below the tolerance, nice!
                return Solution(x, system.var_idxs)
            elif k >= self.maxit:
                # Too many iterations :(
                warnings.warn("The maximum number of iterations was exceeded")
                return Solution(x, system.var_idxs)
            
            x = np.asmatrix(solve(DpL, mU * x + b)) # DpL is lower triangular, so this is fast
            if self.sparse:
                x = x.T # Sparse library transposes the solution... wth scipy

class SuccessiveOverrelaxationSolver:
    def __init__(self, sparse=False, x0=None, tol_type=Tolerance.RESIDUAL, tol=1e-7, maxit=100, ω=2.0):
        # Initialize a SuccessiveOverrelaxationSolver
        # If sparse, convert matrices to sparse before computation
        # x0 is the default starting vector, defaults to 0
        # Given tolerance type
        # tol is the tolerance level given tolerance type
        # maxit is the maximum number of iterations
        # ω is the SOR hyperparameter
        self.sparse = sparse
        self.x0 = x0
        self.tol_type = tol_type
        self.tol = tol
        self.maxit = maxit
        self.ω = ω

    def solve(self, system, metadata=False):
        # Solve system using SOR
        # Return solution x and pass in any metadata to optional metadata argument
        A = system.A
        b = system.b
        x = self.x0

        if self.sparse:
            A = csr_matrix(A)
            D = diags(A.diagonal()) # diagonal component of A
            L = A - triu(A) # strictly lower triangular
            U = A - tril(A) # strictly upper triangular
            solve = spsolve
        else:
            D = np.diag(np.diag(A)) # diagonal component of A
            L = A - np.triu(A)
            U = A - np.tril(A)
            solve = np.linalg.solve

        DpOL = 1 / self.ω * (D + self.ω * L)
        DmOU = 1 / self.ω * ((1 - self.ω) * D - self.ω * U) # precalculate these

        if x is None:
            x = np.zeros((A.shape[1], 1), dtype=np.float64)

        # Begin iteration
        for k in naturals():
            r = b - A * x
            if metadata is not None:
                metadata.save(x=x, r=r)

            if self.tol_type == Tolerance.RESIDUAL and np.linalg.norm(r) <= self.tol:
                # We are below the tolerance, nice!
                return Solution(x, system.var_idxs)
            elif self.tol_type == Tolerance.RELATIVE_RESIDUAL and np.linalg.norm(r) / np.linalg.norm(b) <= self.tol:
                # We are below the tolerance, nice!
                return Solution(x, system.var_idxs)
            elif k >= self.maxit:
                # Too many iterations :(
                warnings.warn("The maximum number of iterations was exceeded")
                return Solution(x, system.var_idxs)
            
            x = np.asmatrix(solve(DpOL, DmOU * x + b)) # DpOL is lower triangular, so this is fast
            if self.sparse:
                x = x.T # Sparse library transposes the solution... wth scipy