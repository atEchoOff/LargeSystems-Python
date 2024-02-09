class ShiftedList:
    # A simple list which indexes by a custom shift
    def __init__(self, start, list, hard=False):
        # If hard, dont allow negative indexing
        self.list = list
        self.start = start
        self.hard = hard

    def __getitem__(self, idx):
        if idx - self.start < 0:
            raise IndexError("This shifted list is hard, it cannot be accessed out of bounds")
        return self.list[idx - self.start]
    
    def __iter__(self):
        self.iter = 0
        return self
    
    def __next__(self):
        if self.iter < len(self.list):
            ret = self.list[self.iter]
            self.iter += 1
            return ret
        else:
            raise StopIteration