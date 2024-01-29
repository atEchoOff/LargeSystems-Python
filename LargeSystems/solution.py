class Solution:
    # A simple vector which can be indexed by multiple indices
    def __init__(self, values, var_idxs):
        # Contain the values along with the index belonging to each variable name
        self.values = values
        self.var_idxs = var_idxs

    def __getitem__(self, key):
        if isinstance(key, list):
            # Return a list of values indexed by the elements of key
            ret = []
            for var in key:
                ret.append(self.values[self.var_idxs[var]])

            return ret
        
        # Otherwise, assume this is just a variable
        return self.values[self.var_idxs[key]]