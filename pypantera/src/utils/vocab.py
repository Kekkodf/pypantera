import os
import numpy as np
import multiprocessing as mp

class Vocab():
    def __init__(self, embPath: str) -> None:
        self.loadParallel(embPath)

    def loadParallel(self, embPath: str) -> None:
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
            results = pool.map(func=self.loadChunk, iterable=[(embPath, i * chunkSize, (i+1) * chunkSize) for i in range(num_processes)])

        # Combine the results from all processes
        self.embeddings = {}
        for result in results:
            self.embeddings.update(result)

    def loadChunk(self, args) -> dict:
        embPath, start, end = args
        embeddings = {}

        with open(embPath, 'r') as file:
            if start != 0:
                file.seek(start)
                file.readline()  # Skip the first line as it may be incomplete
            while file.tell() < end:
                line = file.readline()
                if not line:
                    break  # End of file
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings[word] = vector

        return embeddings
