from numbers import Number
import numpy as np
from copy import deepcopy

class V:
    # A container for variables
    def __init__(self, *vars):
        self.vars = vars

    def __rmul__(self, val):
        # A matrix is being multiplied on the left, this should turn into a linear type
        return Linear(val, self.vars)

class Linear:
    # A builder for the left hand side of a linear equation
    def __init__(self):
        # An empty constructor
        # Should only be used if left and right will be set immediately after, manually
        self.left = None
        self.right = None

        self.RHS = None

    def __init__(self, left=None, right=None):
        # Constructor given left and right parameters, a matrix and some variable names respectively. 
        if left is not None and right is not None:
            if isinstance(left, Number):
                # Shorthand for number times identity matrix, set the correct form
                left = left * np.asmatrix(np.identity(len(right)))

            # Save the matrix and the variables for it
            self.left = [left]
            self.right = [right]
        else:
            self.left = None
            self.right = None
        
        self.RHS = None

    def deepcopy(self):
        # Return a deep copy of self
        ret = Linear()
        ret.left = deepcopy(self.left)
        ret.right = deepcopy(self.right)
        ret.RHS = deepcopy(self.RHS)
        return ret
    
    def __eq__(self, RHS):
        # This is not an equals method!
        # This instead saves the value, right hand side, as the right hand side for this object
        if isinstance(RHS, Number):
            # Shorthand, make the RHS be the correct height
            # First, determine the height
            height = np.shape(self.left[0])[0]

            # Now, set RHS accordingly
            RHS = [[RHS]] * height

        self.RHS = RHS
        return self

    def __mul__(self, val):
        # Multiply all matrices by some value
        ret = deepcopy(self)
        for i in range(0, len(self.left)):
            ret.left[i] = val * self.left[i]

        return ret

    def __rmul__(self, val):
        # Multiply all matrices by some value
        ret = deepcopy(self)
        for i in range(0, len(self.left)):
            ret.left[i] = self.left[i] * val

        return ret
    
    def __imul__(self, val):
        # Do the same as __mul__ but modify self
        for i in range(0, len(self.left)):
            self.left[i] = val * self.left[i]

        return self

    def __add__(self, val):
        # Add a linear element to another
        # Note, this is not to be used for adding matrices or scalars to elements
        # This is a "linear object", to add constants makes it affine
        ret = deepcopy(self)
        ret.left.extend(val.left)
        ret.right.extend(val.right)

        return ret
    
    def __radd__(self, val):
        # Add a linear element to the left of this one
        # See description for __add__
        ret = deepcopy(val)
        ret.left.extend(self.left)
        ret.right.extend(self.right)

        return ret
    
    def __iadd__(self, val):
        # Same as __add__, but modify self
        self.left.extend(val.left)
        self.right.extend(val.right)

        return self
    
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
    
    def T(self):
        # Transpose all coefficient matrices
        ret = deepcopy(self)
        for i in range(0, len(self.left)):
            ret.left[i] = self.left[i].T

        return ret
    
ZERO = Linear([], [])