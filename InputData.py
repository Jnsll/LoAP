import pandas as pd

class InputData:

    def __init__(self, path, file_name):
        self.features = []
        self.path = path
        self.file_name = file_name
        self.data = pd.DataFrame()
        
    
    def retrieve_execution_time(self):
        
