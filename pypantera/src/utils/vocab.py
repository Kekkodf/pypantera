import os
import numpy as np
import multiprocessing as mp

class Vocab():
    def __init__(self, embPath: str) -> None:
        '''
        Vocabulary class: this class is used to define the vocabulary of the embeddings

        To optimize the loading of the embeddings, the loading is done in parallel using the multiprocessing library and the load_parallel method
        '''
        self.load_parallel(embPath)

    def load_parallel(self, embPath: str) -> None:
        '''
        # Load Parallel

        method load_parallel: this method is used to load the embeddings in parallel

        It sets the embeddings attribute of the class

        : param embPath: str the path to the embeddings file
        : return: None

        Usage example:

        >>> vocab = Vocab('path/to/embeddings.txt')
        >>> vocab.embeddings

        (Method used in the __init__ method of the mechansims classes)
        '''
        if not os.path.exists(embPath):
            raise FileNotFoundError(f"File {embPath} not found")

        # Determine the number of processes based on the available CPUs
        num_processes = mp.cpu_count()

        # Determine the size of each chunk to be processed by each process
        fileSize = os.path.getsize(embPath)
        chunkSize = fileSize // num_processes

        # Create a pool of processes
        with mp.Pool(processes=num_processes) as pool:
            # Map each chunk to a separate process
            results = pool.map(func=self._load_chunk, iterable=[(embPath, i * chunkSize, (i+1) * chunkSize) for i in range(num_processes)])

        # Combine the results from all processes
        self.embeddings = {}
        for result in results:
            self.embeddings.update(result)

    def _load_chunk(self, args) -> dict:
        '''
        chunks and loads the embeddings in parallel
        '''
        embPath, start, end = args
        embeddings = {}

        with open(embPath, 'r') as file:
            if start != 0:
                file.seek(start)
                file.readline() 
            while file.tell() < end:
                line = file.readline()
                if not line:
                    break  
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings[word] = vector

        return embeddings
