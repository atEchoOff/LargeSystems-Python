class Layered:
    # A container which has a main section and an outer section
    # At construction, initialize the main part of the list
    # User can then set more values outside of the list's bounds
    # This is useful for creation of variables for system and setting boundary values outside

    def __init__(self, list):
        self.list = list
        self.outer = dict()

    def __getitem__(self, idxs):
        # If index is in list, return that. Otherwise, try to return element from outer
        try:
            # Treat vec[i,j] as vec[i][j] etc
            # This is different than julia which has a built-in multi type matrix type
            ret = self.list
            for idx in idxs:
                ret = ret[idx]
            return ret
        except:
            return self.outer[idxs]
        
    def __setitem__(self, key, value):
        self.outer[key] = value