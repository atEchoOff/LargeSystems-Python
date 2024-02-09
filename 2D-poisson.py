from LargeSystems.solvers import SimpleSolver
import matplotlib.pyplot as plt
from Utils.PDEs import build_2D_poisson
from Utils.ShiftedList import ShiftedList
import numpy as np

# We will model the PDE -u''(x)=f(x)

f = lambda x, y: ((y - y**3)*(2 * np.pi)**2 + 6*y) * np.cos(2*np.pi * x) - 6*y

u = lambda x, y: (y**3 - y) * (np.cos(2*np.pi * x) - 1)
boundary = lambda x, y: 0 # our boundary value function

def evaluate_2D_poisson(N):
    # Evaluate the ODE, and then return the L2 error (modified for PDEs)
    h = 1 / (N + 1)
    system = build_2D_poisson(N, boundary, f, stencil=9)
    solver = SimpleSolver(sparse=True)

    solution = solver.solve(system)
    U = np.asmatrix([[solution[f"U{i},{j}"] for j in range(1, N + 1)] for i in range(1, N + 1)])
    true_U = np.asmatrix([[u(h * x, h * y) for x in range(1, N + 1)] for y in range(1, N + 1)])

    return h * np.linalg.norm(U - true_U, "fro") # the frobenius norm

def estimate_numerical_convergence(Ns, errors):
    # Estimate the c and n for numerical convergence, based on a vector of Ns and errors
    Ns = np.array(Ns)
    hs = np.log(1 / (1 + Ns))
    errors = np.log(errors)

    n, log_c = np.polyfit(hs, errors, 1)
    return np.exp(log_c), n

def evaluate_Ns(*Ns):
    # Evaluate the PDE for different Ns, get the L2 error, find the convergence rates and plot
    L2_errors = []
    for N in Ns:
        L2_errors.append(evaluate_2D_poisson(N))
    
    c, n = estimate_numerical_convergence([*Ns], L2_errors)
    print(c, n)

evaluate_Ns(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)