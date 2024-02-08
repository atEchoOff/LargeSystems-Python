class ShiftedList:
    # A simple list which indexes by a custom shift
    def __init__(self, start, list):
        self.list = list
        self.start = start

    def __getitem__(self, idx):
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