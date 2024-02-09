from Utils.ShiftedList import ShiftedList
from LargeSystems.linear import V
from LargeSystems.system import System
from Utils.Layered import Layered

def build_1D_poisson(N, a, b, f):
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

def build_2D_poisson(N, boundary, f, Δf=lambda x, y:0, stencil=9):
    # Build the PDE Δu = f(x,y)
    # So that over ∂([0,1]²), u(x,y)=boundary(x,y)
    # Optionally pass the laplacian of f to improve convergence to O(h⁴), works only when stencil=9
    # Choose either the 5 point or 9 point stencil

    h = 1 / (N + 1)
    U = [[f"U{i},{j}" for i in range(1, N + 1)] for j in range(1, N + 1)]
    system = System(U)
    U = Layered(ShiftedList(1, 
                [ShiftedList(1, 
                    [V(f"U{i},{j}") for i in range(1, N + 1)]) 
                    for j in range(1, N + 1)])) # Blame python, not me
    
    # Set boundary values
    for i in range(0, N + 2):
        U[i,0] = boundary(i * h, 0)
        U[i, N + 1] = boundary(i * h, 1)
    
    for j in range(0, N + 2):
        U[0,j] = boundary(0, j * h)
        U[N + 1,j] = boundary(1, j * h)

    # Add constraints
    if stencil == 9:
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                system.add_constraint(1 / (6*h**2) * (4*U[i-1,j] + 4*U[i+1,j] + 4*U[i,j-1] 
                                        + 4*U[i,j+1] + U[i-1,j-1] + U[i+1,j-1] + U[i+1,j+1] + U[i-1,j+1] - 20*U[i,j]) == f(h*i, h*j) + h**2 / 12 * Δf(h*i, h*j))
            
    elif stencil == 5:
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                system.add_constraint(1 / h**2 * (U[i-1,j] + U[i+1,j] + U[i,j-1] + U[i,j+1] - 4*U[i,j]) == f(h*i, h*j))


    return system