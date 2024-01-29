from LargeSystems.systembuilder import SystemBuilder
from LargeSystems.linear import V
from ShiftedList import *


class TMOCSolver:
    # A class designed for solving an optimal control problem of the form given in assignment 2
    # Using the theta method, with equal stepsizes h
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
        self._lambda = ShiftedList(1, [ShiftedList(1, ["lambda" + str(i) + str(j) for j in range(1, ny + 1)]) for i in range(1, K + 1)])
        self._u = ShiftedList(0, [ShiftedList(1, ["u" + str(i) + str(j) for j in range(1, nu + 1)]) for i in range(0, K + 1)])

        self.grad_lf = None

        self.grad_y_l = None
        self.grad_u_l = None

        self.f = None
        self.jacobi_f_y = None
        self.jacobi_f_u = None
    
    def with_grad_lf(self, grad_lf):
        self.grad_lf = grad_lf
        return self
    
    def with_grad_y_l(self, grad_y_l):
        self.grad_y_l = grad_y_l
        return self
    
    def with_grad_u_l(self, grad_u_l):
        self.grad_u_l = grad_u_l
        return self
    
    def with_f(self, f):
        self.f = f
        return self
    
    def with_jacobi_f_y(self, jacobi_f_y):
        self.jacobi_f_y = jacobi_f_y
        return self
    
    def with_jacobi_f_u(self, jacobi_f_u):
        self.jacobi_f_u = jacobi_f_u
        return self
    
    def solve(self, theta):
        # Solve the problem given theta
        # A Solution object is returned

        # First, make sure we have everything from the builder pattern
        if self.grad_lf == None:
            print("grad_lf was not initialized")
            return
        
        if self.grad_y_l == None:
            print("grad_y_l was not initialized")
            return
        
        if self.grad_u_l == None:
            print("grad_u_l was not initialized")
            return
        
        if self.f == None:
            print("f was not initialized")
            return
        
        if self.jacobi_f_y == None:
            print("jacobi_f_y was not initialized")
            return
        
        if self.jacobi_f_u == None:
            print("jacobi_f_u was not initialized")
            return
        
        # Collect the variables into one long list
        all_vars = []
        for y in self._y.list:
            all_vars.extend(y.list)
        
        for l in self._lambda.list:
            all_vars.extend(l.list)

        for u in self._u.list:
            all_vars.extend(u.list)

        # Shorthand
        K = self.K
        y = ShiftedList(0, [V(*self._y[i].list) for i in range(0, K + 1)])
        _lambda = ShiftedList(1, [V(*self._lambda[i].list) for i in range(1, K + 1)])
        u = ShiftedList(0, [V(*self._u[i].list) for i in range(0, K + 1)])
        h = self.h

        t = [self.t0 + h * i for i in range(0, K + 1)]
        
        builder = SystemBuilder(all_vars)

        # First, add the lambda_k constraint
        builder.add_constraint(0 == 1 * _lambda[K] + self.grad_lf(y[K])
                                    + h * (1 - theta) * self.grad_y_l(y[K], u[K], t[K])\
                                    - h * (1 - theta) * self.jacobi_f_y.T * _lambda[K])

        for k in range(0, K - 1):
            # Add constraint from (9)
            builder.add_constraint(0 == 1 * _lambda[k + 1]
                                        - h * (1 - theta) * self.jacobi_f_y.T * _lambda[k + 1]
                                        + h * self.grad_y_l(y[k + 1], u[k + 1], t[k + 1])
                                        - 1 * _lambda[k + 2]
                                        - h * theta * self.jacobi_f_y.T * _lambda[k + 2])
            
        
        # Add the constraint from (10)
        builder.add_constraint(0 == h * theta * self.grad_u_l(y[0], u[0], t[0])
                                    - h * theta * self.jacobi_f_u.T * _lambda[1])
        
        # Add the constraint from (11)
        builder.add_constraint(0 == h * (1 - theta) * self.grad_u_l(y[K], u[K], t[K])
                                    - h * (1 - theta) * self.jacobi_f_u.T * _lambda[K])
        
        for k in range(1, K):
            # Add the constraint from (12)
            builder.add_constraint(0 == h * self.grad_u_l(y[k], u[k], t[k])
                                        - h * theta * self.jacobi_f_u.T * _lambda[k + 1]
                                        - h * (1 - theta) * self.jacobi_f_u.T * _lambda[k])
            
        # Add theta method base case
        builder.add_constraint(self.y0 == 1 * y[0])

        # Add all theta method steps
        for k in range(0, K):
            builder.add_constraint(0 == 1 * y[k] - 1 * y[k + 1]
                                        + h * theta * self.f(y[k], u[k], t[k]) 
                                        + h * (1 - theta) * self.f(y[k + 1], u[k + 1], t[k + 1]))
        
        return builder.solve()