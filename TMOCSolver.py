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
        K = self.K
        y = ShiftedList(0, [V(*self._y[i].list) for i in range(0, K + 1)])
        λ = ShiftedList(1, [V(*self.λ[i].list) for i in range(1, K + 1)])
        u = ShiftedList(0, [V(*self._u[i].list) for i in range(0, K + 1)])
        h = self.h

        t = [self.t0 + h * i for i in range(0, K + 1)]
        
        builder = SystemBuilder(all_vars)

        # First, add the lambda_k constraint
        builder.add_constraint(0 == λ[K] + self.Δlᶠ(y[K])
                                    + h * (1 - θ) * self.Δᵧl(y[K], u[K], t[K])\
                                    - h * (1 - θ) * self.fᵧ.T * λ[K])

        for k in range(0, K - 1):
            # Add constraint from (9)
            builder.add_constraint(0 == λ[k + 1]
                                        - h * (1 - θ) * self.fᵧ.T * λ[k + 1]
                                        + h * self.Δᵧl(y[k + 1], u[k + 1], t[k + 1])
                                        - λ[k + 2]
                                        - h * θ * self.fᵧ.T * λ[k + 2])
            
        
        # Add the constraint from (10)
        builder.add_constraint(0 == h * θ * self.Δᵤl(y[0], u[0], t[0])
                                    - h * θ * self.fᵤ.T * λ[1])
        
        # Add the constraint from (11)
        builder.add_constraint(0 == h * (1 - θ) * self.Δᵤl(y[K], u[K], t[K])
                                    - h * (1 - θ) * self.fᵤ.T * λ[K])
        
        for k in range(1, K):
            # Add the constraint from (12)
            builder.add_constraint(0 == h * self.Δᵤl(y[k], u[k], t[k])
                                        - h * θ * self.fᵤ.T * λ[k + 1]
                                        - h * (1 - θ) * self.fᵤ.T * λ[k])
            
        # Add θ method base case
        builder.add_constraint(self.y0 == y[0])

        # Add all θ method steps
        for k in range(0, K):
            builder.add_constraint(0 == y[k] - y[k + 1]
                                        + h * θ * self.f(y[k], u[k], t[k]) 
                                        + h * (1 - θ) * self.f(y[k + 1], u[k + 1], t[k + 1]))
        
        return builder.solve()