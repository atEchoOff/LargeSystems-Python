from numbers import Number
import numpy as np
from copy import deepcopy

class Identity:
    # An identity matrix simulator for linear systems
    def __init__(self, val, dim):
        self.val = val
        self.shape = [dim]

    def __getitem__(self, keys):
        # Simulate getting a value from the identity matrix, dont check bounds
        # We only use these matrices to get column arrays anyway
        i, j = keys
        
        if isinstance(i, slice) and i.start == None and i.stop == None and i.step == None:
            # self[:,index]
            ret = np.zeros((self.shape[0], 1))
            ret[j, 0] = self.val
            return ret
        
        else:
            raise AttributeError("Can only get columns from identity matrix")
    
    def __mul__(self, other):
        # Multiply by identity matrix
        if isinstance(other, Number):
            # Instead return a new identity which contains a new number in the diagonal
            return Identity(self.val * other, self.shape[0])
        
        # Instead, assume we are multiplying a matrix or vector
        return self.val * other
    
    def __rmul__(self, other):
        # Multiply by identity matrix
        if isinstance(other, Number):
            # Instead return a new identity which contains a new number in the diagonal
            return Identity(other * self.val, self.shape[0])
        
        # Instead, assume we are multiplying a matrix or vector
        return other * self.val
    
    def __truediv__(self, other):
        # Divide identity by float
        return self * (1 / other)
    
    def __neg__(self):
        return Identity(-self.val, self.shape[0])

def V(*vars):
    # Convert variables to a linear system
    # Items passed in as a series of lists
    # If one list, treated as one variable
    # If multiple, return as a list of variables
    if len(vars) == 1:
        return Linear(Identity(1, len(vars[0])), vars[0])
    
    ret = []
    for var in vars:
        if isinstance(var, str):
            # variable must be packed in a list
            var = [var]
        ret.append(Linear(Identity(1, len(var)), var))
    return ret

class Linear:
    # A builder for the left hand side of a linear equation
    def __init__(self, left=None, right=None):
        # Constructor given left and right parameters, a matrix and some variable names respectively. 
        if left is not None and right is not None:
            # Save the matrix and the variables for it
            self.left = [left]
            self.right = [right]
            self.constant = np.zeros((left.shape[0], 1), np.float64)
        else:
            self.left = None
            self.right = None

    def deepcopy(self):
        # Return a deep copy of self
        ret = Linear()
        ret.left = deepcopy(self.left)
        ret.right = deepcopy(self.right)
        ret.constant = deepcopy(self.constant)
        return ret

    def __eq__(self, RHS):
        # This is not an equals method!
        # This instead saves the value, right hand side, as the right hand side for this object
        # Return an equation object
        if isinstance(RHS, Number):
            # Shorthand, make the RHS be the correct height
            # First, determine the height
            height = self.left[0].shape[0]

            # Now, set RHS accordingly
            RHS = np.asmatrix([[RHS]] * height)

            return Equation(self, RHS - self.constant)
        
        if isinstance(RHS, list):
            # Set equal to the list directly
            return Equation(self, np.asmatrix(RHS) - self.constant)
        
        if isinstance(RHS, np.matrix):
            return Equation(self, RHS - self.constant)
        
        if isinstance(RHS, Linear):
            # We have two linear systems, assume we are on the left
            left = self - RHS
            return Equation(left, RHS.constant - self.constant)

        print("ERROR: EQUALITY BETWEEN LINEAR AND " + type(RHS) + " IS NOT DEFINED")
        return None

    def __mul__(self, val):
        # Multiply all matrices by some value
        ret = deepcopy(self)
        for i in range(0, len(self.left)):
            ret.left[i] = self.left[i] * val

        ret.constant = self.constant * val

        return ret

    def __rmul__(self, val):
        # Multiply all matrices by some value
        ret = deepcopy(self)
        for i in range(0, len(self.left)):
            ret.left[i] = val * self.left[i]

        ret.constant = val * self.constant

        return ret
    
    def __truediv__(self, other):
        # Divide self by some float
        return self * (1 / other)

    def __add__(self, val):
        # Add a linear element to another
        if isinstance(val, Number):
            # Shorthand, replace val with a numpy matrix
            val = np.asmatrix([[val]] * self.left[0].shape[0])

        if isinstance(val, np.matrix):
            # Add to constant directly
            ret = deepcopy(self)
            ret.constant = self.constant + val

            return ret
        
        # Assume we are adding a linear
        ret = deepcopy(self)
        ret.left.extend(val.left)
        ret.right.extend(val.right)
        ret.constant = self.constant + val.constant

        return ret
    
    def __radd__(self, val):
        # Add a linear element to the left of this one
        # See description for __add__
        if isinstance(val, Number):
            # Shorthand, replace val with a numpy matrix
            val = np.asmatrix([[val]] * self.left[0].shape[0])

        if isinstance(val, np.matrix):
            # Add to constant directly
            ret = deepcopy(self)
            ret.constant = val + self.constant

            return ret
        
        # Assume we are adding a linear
        ret = deepcopy(val)
        ret.left.extend(self.left)
        ret.right.extend(self.right)
        ret.constant = val.constant + self.constant

        return ret
    
    def __neg__(self):
        # Return -1 times all matrices in this system
        return -1 * self
    
    def __sub__(self, val):
        # Add the negative of a linear system
        return self + (-val)
    
    def __rsub__(self, val):
        # Add the negative of this system to another
        return val + (-self)
    
    def __isub__(self, val):
        # Add the negative of a linear system, in place
        self = self + (-val)
        return self
    
class Equation:
    # Stores a linear equation, with the left hand side and the right hand side!
    # Really the only point of this class is to pass into a systembuilder
    # and not have any operators
    def __init__(self, linear, RHS):
        self.linear = linear
        self.RHS = RHS