from Utils.ShiftedList import ShiftedList
from LargeSystems.linear import V
from LargeSystems.system import System

def build_poisson(N, a, b, f):
    # Build PDE -u''(x) = f(x)
    # So that u(0) = a
    # and u(1) = b

    h = 1 / (N + 1)
    U = [f"U{i}" for i in range(1, N + 1)]
    system = System(U)
    U = ShiftedList(1, V(*U)) # Turn U into a list of variables for indexing
    
    # Add our constraints
    system.add_constraint((2 * U[1] - U[2]) / h**2 == f(h * 1) + a / h**2)
    for i in range(2, N):
        system.add_constraint((-U[i - 1] + 2 * U[i] - U[i + 1]) / h**2 == f(h * i))

    system.add_constraint((-U[N - 1] + 2 * U[N]) / h**2 == f(h * N) + b / h**2)

    return system