import scipy
from pathlib import Path

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read_mtx(self):
        file = Path(self.file_path)
        if file.is_file() and file.suffix=='.mtx':
            return scipy.io.mmread(self.file_path)
        else:
            print("Can't locate the file")
            if not file.endswith('.mtx'):
                print("Data format is not .mtx")
