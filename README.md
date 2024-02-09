Ever have a massive system of equations where the variables are complicated and dynamic? Ever try to make a massive system for an optimal control problem and get stuck trying to figure out what indices go where in your massive 100x100 matrix? Worry no longer. This Python library (with a twin Julia library) is designed for exactly those purposes: the creation of linear systems of equations using a simple and easy notation. See below, a quick demo:

    from LargeSystems.linear import V
    from LargeSystems.solvers import SimpleSolver
    from LargeSystems.system import System
    
    x, y, z = V("x", "y", "z")
    system = System("x", "y", "z")
    
    system.add_constraint(2 * x + 3 * y - z == 2)
    system.add_constraint(2 * x - z + y == 1)
    system.add_constraint(x + y + z == 1)
    
    solver = SimpleSolver()
    print(solver.solve(system).values)

See how clean that is?? Three constraints, easily plugged in to a 3x3 matrix, without having to worry about indexing. However, the system above is very simple. Want to get more complicated? What about the Laplace 9-point stencil matrix? Each row follows the constraint

![9 Point Laplacian Stencil](https://www.dropbox.com/scl/fi/8md9qd1on8nq084kuycxf/Laplacian.png?rlkey=8srlx3ogjag88qd2wu6hake4r&raw=1)

Where $i$ and $j$ vary from 1 to $N$. Also, there's boundary conditions that need to be taken into account! Usually, this matrix is massive, taking account for the equations on the corners, each side of a square, and all the internal points, and keeping track of it can be a nightmare! Even in MATLAB, where there are functions to put together these matrices, the resulting code is borderline unreadable. However, below, you can see that the equations are plugged directly in:

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

See how beautifully readable and concise that is? The boundary values are set, and then every single row is taken care of in a single equation! Okay okay I hear you, your system is more complicated than the simple 9-point stencil above, a notoriously annoying matrix to build. No, you need the real deal. Consider the theta method optimal control problem, whose equations are goverened by:

![Theta Method Optimal Control System](https://www.dropbox.com/scl/fi/kvelfn6ssnnprhgowwam9/TMOC.png?rlkey=wtfkheal1fx8u0w5tmhjj29iv&raw=1)

Woof! Needless to say, creating the matrix for the system above should require hundreds of lines of code. This library? Plug the constraints directly in:

    # Start building our system!
    system = System(_y, _λ, _u)
    
    # Constraint from (8)
    system.add_constraint(λ[K] - h * (1 - θ) * fᵧ.T * λ[K] ==
                                -Δlᶠ(y[K]) - h * (1 - θ) * Δᵧl(y[K], u[K], t[K]))
    
    for k in range(0, K - 1):
        # Add constraint from (9)
        system.add_constraint(λ[k + 1] - h * (1 - θ) * fᵧ.T * λ[k + 1] ==
                                    -h * Δᵧl(y[k + 1], u[k + 1], t[k + 1]) + λ[k + 2]
                                    + h * θ * fᵧ.T * λ[k + 2])
        
    
    # Add the constraint from (10)
    system.add_constraint(0 == h * Δᵤl(y[0], u[0], t[0])
                                - h * fᵤ.T * λ[1])
    
    # Add the constraint from (11)
    system.add_constraint(0 == h * Δᵤl(y[K], u[K], t[K])
                                - h * fᵤ.T * λ[K])
    
    for k in range(1, K):
        # Add the constraint from (12)
        system.add_constraint(0 == h * Δᵤl(y[k], u[k], t[k])
                                    - h * θ * fᵤ.T * λ[k + 1]
                                    - h * (1 - θ) * fᵤ.T * λ[k])
        
    # Add θ method base case
    system.add_constraint(y[0] == y0)
    
    # Add all θ method steps
    for k in range(0, K):
        system.add_constraint(y[k + 1] == y[k]
                                    + h * θ * f(y[k], u[k], t[k]) 
                                    + h * (1 - θ) * f(y[k + 1], u[k + 1], t[k + 1]))
    
    return SimpleSolver(sparse=True).solve(system) # The resulting system is relative sparse

Okay needless to say, this still looks very complicated, but think of how much more simple it is with this library! I rest my case. 

You dont like python? You are more of a julia lover? Check out my equivalent julia library.
