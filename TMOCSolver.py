from LargeSystems.systembuilder import SystemBuilder
from LargeSystems.linear import V
from ShiftedList import *


class TMOCSolver:
    # A class designed for solving an optimal control problem of the form given in assignment 2
    # Using the θ method, with equal stepsizes h
    # Due to the amount of parameters, this class will use the builder construct

    def __init__(self, t0, y0, h, K, ny, nu):
        self.t0 = t0
        self.y0 = y0
        self.h = h
        self.K = K
        self.ny = ny
        self.nu = nu

        # Initialize the variables
        self._y = ShiftedList(0, [ShiftedList(1, ["y" + str(i) + str(j) for j in range(1, ny + 1)]) for i in range(0, K + 1)])
        self.λ = ShiftedList(1, [ShiftedList(1, ["λ" + str(i) + str(j) for j in range(1, ny + 1)]) for i in range(1, K + 1)])
        self._u = ShiftedList(0, [ShiftedList(1, ["u" + str(i) + str(j) for j in range(1, nu + 1)]) for i in range(0, K + 1)])

        self.Δlᶠ = None

        self.Δᵧl = None
        self.Δᵤl = None

        self.f = None
        self.fᵧ = None
        self.fᵤ = None
    
    def with_Δlᶠ(self, Δlᶠ):
        self.Δlᶠ = Δlᶠ
        return self
    
    def with_Δᵧl(self, Δᵧl):
        self.Δᵧl = Δᵧl
        return self
    
    def with_Δᵤl(self, Δᵤl):
        self.Δᵤl = Δᵤl
        return self
    
    def with_f(self, f):
        self.f = f
        return self
    
    def with_fᵧ(self, fᵧ):
        self.fᵧ = fᵧ
        return self
    
    def with_fᵤ(self, fᵤ):
        self.fᵤ = fᵤ
        return self
    
    def solve(self, θ):
        # Solve the problem given θ
        # A Solution object is returned

        # First, make sure we have everything from the builder pattern
        if self.Δlᶠ == None:
            print("Δlᶠ was not initialized")
            return
        
        if self.Δᵧl == None:
            print("Δᵧl was not initialized")
            return
        
        if self.Δᵤl == None:
            print("Δᵤl was not initialized")
            return
        
        if self.f == None:
            print("f was not initialized")
            return
        
        if self.fᵧ == None:
            print("fᵧ was not initialized")
            return
        
        if self.fᵤ == None:
            print("fᵤ was not initialized")
            return
        
        # Collect the variables into one long list
        all_vars = []
        for y in self._y.list:
            all_vars.extend(y.list)
        
        for l in self.λ.list:
            all_vars.extend(l.list)

        for u in self._u.list:
            all_vars.extend(u.list)

        # Shorthand
        Δlᶠ = self.Δlᶠ
        Δᵧl = self.Δᵧl
        Δᵤl = self.Δᵤl
        f = self.f
        fᵧ = self.fᵧ
        fᵤ = self.fᵤ
        K = self.K
        y = ShiftedList(0, [V(*self._y[i].list) for i in range(0, K + 1)])
        λ = ShiftedList(1, [V(*self.λ[i].list) for i in range(1, K + 1)])
        u = ShiftedList(0, [V(*self._u[i].list) for i in range(0, K + 1)])
        h = self.h
        y0 = self.y0

        # Initialize our domain
        t = [self.t0 + h * i for i in range(0, K + 1)]
        
        builder = SystemBuilder(all_vars)

        # First, add the lambda_k constraint
        builder.add_constraint(λ[K] - h * (1 - θ) * fᵧ.T * λ[K] ==
                                    -Δlᶠ(y[K]) - h * (1 - θ) * Δᵧl(y[K], u[K], t[K]))

        for k in range(0, K - 1):
            # Add constraint from (9)
            builder.add_constraint(λ[k + 1] - h * (1 - θ) * fᵧ.T * λ[k + 1] ==
                                        -h * Δᵧl(y[k + 1], u[k + 1], t[k + 1]) + λ[k + 2]
                                        + h * θ * fᵧ.T * λ[k + 2])
            
        
        # Add the constraint from (10)
        builder.add_constraint(0 == h * θ * Δᵤl(y[0], u[0], t[0])
                                    - h * θ * fᵤ.T * λ[1])
        
        # Add the constraint from (11)
        builder.add_constraint(0 == h * (1 - θ) * Δᵤl(y[K], u[K], t[K])
                                    - h * (1 - θ) * fᵤ.T * λ[K])
        
        for k in range(1, K):
            # Add the constraint from (12)
            builder.add_constraint(0 == h * Δᵤl(y[k], u[k], t[k])
                                        - h * θ * fᵤ.T * λ[k + 1]
                                        - h * (1 - θ) * fᵤ.T * λ[k])
            
        # Add θ method base case
        builder.add_constraint(y[0] == y0)

        # Add all θ method steps
        for k in range(0, K):
            builder.add_constraint(y[k + 1] == y[k]
                                        + h * θ * f(y[k], u[k], t[k]) 
                                        + h * (1 - θ) * f(y[k + 1], u[k + 1], t[k + 1]))
        
        return builder.solve()