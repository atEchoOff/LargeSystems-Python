import numpy as np

class SparseSystem:
    # An interface for sparse solvers
    # This impliments a basic scheme where the system stores linear systems
    # Systems should use the linears directly to solve their system
    def __init__(self):
        self.linears = []

    def evaluate(self, **vars):
        # Evaluate all linear terms in system
        tot_height = sum(self.linears[i].linear.left[0].shape[0] for i in range(0, len(self.linears)))

        ret = np.zeros((tot_height, 1))

        determined = 0
        for i in range(0, len(self.linears)):
            linear = self.linears[i].linear
            ret[determined:determined + linear.left[0].shape[0],0] = linear.evaluate(**vars)

            determined += linear.left[0].shape[0]

        return ret

    def add_constraint(self, equation):
        # Add an equation to the system
        self.linears.append(equation)

    def solve(self):
        raise NotImplementedError("Choose a specific system solver, this is the interface")