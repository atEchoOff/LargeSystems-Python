import numpy as np
from LargeSystems.solution import *
from LargeSystems.linear import *

class SystemBuilder:
    # A tool to build large linear systems of equations
    
    def __init__(self, names):
        # Initialize a system builder given names for each variable
        self.var_names = names
        self.A = np.matrix(np.zeros((len(names), len(names)), np.float64))
        self.b = np.zeros((len(names), 1), np.float64)

        self.var_idxs = dict()

        self.determined = 0

        for i, variable in enumerate(names):
            self.var_idxs[variable] = i

    def add_constraint(self, linear):
        # Accept an equation of the form RHS = A1 * x1 + A2 * x2 + ...
        # Cast RHS to a numpy array
        RHS = np.asmatrix(linear.RHS, dtype=np.float64)

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

    def solve(self):
        # Sovle the build system, return a solution vector
        x = np.linalg.solve(self.A, self.b)

        # Make x flat, and convert to list
        x = x.T.tolist()[0]
        return Solution(x, self.var_idxs)
    
if __name__ == "__main__":
    # Here we can run some tests
    vars = ['a', 'b', 'c']
    builder = SystemBuilder(vars)

    builder.add_constraint(0 == 1 * V("a"))
    builder.add_constraint([[1], [1]] == 1 * V("b", "c"))

    print(builder.A)
    print(builder.b)

    solution = builder.solve()
    print(solution["a"])
    print(solution[["b", "c"]])

    # Make a more complicated system
    x = ["x" + str(i) for i in range(1, 4)]
    y = ["y" + str(i) for i in range(1, 3)]

    vars = x[:]
    vars.extend(y)

    matr1 = np.matrix([
        [0, 2, 0],
        [1, 0, 1]
    ], dtype=np.float64)

    matr2 = np.matrix([
        [0, 1],
        [1, 1]
    ], dtype=np.float64)

    matr3 = np.matrix([
        [1, 2, 3, 4]
    ], dtype=np.float64)

    builder = SystemBuilder(vars)
    builder.add_constraint(0 == matr1 * V(*x) - .05 * matr2 * V(*y))
    builder.add_constraint([[1], [2]] == matr2 * V(*y))
    builder.add_constraint(-1 == matr3 * V("x1", "x2", "x3", "y2"))

    print(builder.A)
    print(builder.b)

    solution = builder.solve()

    print(solution[x])
    print(solution[y])