import os
import numpy as np

class Vocab():
    '''
    Class Vocab: this class is used to load the embeddings from a file
    '''
    def __init__(self, embPath: str) -> None:
        '''
        Initialization of the Vocab Object
        '''

        self.load(embPath)
    
    def load(self, embPath: str) -> None:
        
        '''
        Load method: this method is used to load the embeddings from the file
        : param embPath: str the path to the embeddings file
        '''

        embeddings = {}
        #verify if the file exists
        if not os.path.exists(embPath):
            raise FileNotFoundError(f"File {embPath} not found")
        with open(embPath, 'r') as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings[word] = vector
        self.embeddings = embeddings
