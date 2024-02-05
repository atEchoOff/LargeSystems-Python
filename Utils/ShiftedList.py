class ShiftedList:
    # A simple list which indexes by a custom shift
    def __init__(self, start, list):
        self.list = list
        self.start = start

    def __getitem__(self, idx):
        return self.list[idx - self.start]