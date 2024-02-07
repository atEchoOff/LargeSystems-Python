import numpy as np
from LargeSystems.solution import *
from LargeSystems.linear import *

class System:
    # Build a dense system of the form Ax = b
    # Add linear systems to matrix form as constraints are added
    
    def __init__(self, names):
        # Initialize a system builder given names for each variable
        self.var_names = names
        self.A = np.matrix(np.zeros((len(names), len(names)), np.float64))
        self.b = np.zeros((len(names), 1), np.float64)

        self.var_idxs = dict()

        self.determined = 0

        for i, variable in enumerate(names):
            self.var_idxs[variable] = i

    def add_constraint(self, equation):
        # Accept an equation of the form RHS = A1 * x1 + A2 * x2 + ...
        # Cast RHS to a numpy array
        RHS = np.asmatrix(equation.RHS, dtype=np.float64)

        linear = equation.linear
        height = linear.left[0].shape[0]
        bottom = self.determined + height

        # Add RHS to the corresponding position in b
        self.b[self.determined:bottom] += RHS

        for i in range(0, len(linear.left)):
            # We have a new part of the left hand side, Ai * xi
            newnewA = linear.left[i]
            newnewx = linear.right[i]

            # Here, we loop through each variable and place the corresponding column of Ai into the correct
            # column of newA for that corresponding variable
            for i, var in enumerate(newnewx):
                # Get the column index of the variable
                var_idx = self.var_idxs[var]

                # Add the ith column of new_new_A into the var_idx column of newA
                self.A[self.determined:bottom,var_idx] += newnewA[:,i]
        
        # Now, the equation is synonymous to RHS = newA * vars. Add the newA to the bottom of A, and RHS
        # to the bottom of b
        self.determined = bottom