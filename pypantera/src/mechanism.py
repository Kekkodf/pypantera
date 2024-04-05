from .utils.vocab import Vocab
import numpy as np
import os
from typing import List
import numpy.random as npr
import multiprocessing as mp

class Mechanism():
    def __init__(self, kwargs) -> None:
        self.vocab: Vocab = Vocab(kwargs['embPath']) #load the vocabulary
        self.embMatrix: np.array = np.array(
            list(self.vocab.embeddings.values())
            ) #load the embeddings matrix
        self.index2word: dict = {
            i: word 
            for i, word in enumerate(self.vocab.embeddings.keys())
            } #create the index to word mapping
        self.word2index: dict = {
            word: i 
            for i, word in enumerate(self.vocab.embeddings.keys())
            } #create the word to index mapping
        
    def pullNoise(self) -> np.array:
        '''
        method pullNoise: this method is used to pull noise accordingly 
        to the definition of the CMP mechanism, see BibTeX ref

        : return: np.array the noise pulled

        Usage example:
        (Considering that the Mechanism Object mech1 has been created 
        as in the example of the __init__ method)
        >>> mech1.pullNoise()
        '''
        N: np.array = self.epsilon * npr.multivariate_normal(
            np.zeros(self.embMatrix.shape[1]),
            np.eye(self.embMatrix.shape[1])) #pull noise from a multivariate normal distribution
        X: np.array = N / np.sqrt(np.sum(N ** 2)) #normalize the noise
        Y: np.array = npr.gamma(
            self.embMatrix.shape[1],
            1 / self.epsilon) #pull gamma noise
        Z: np.array = Y * X #compute the final noise
        return Z

    def obfuscateText(self, data: str, numberOfCores: int) -> List[str]:
        '''
        method obfuscateText: this method is used to obfuscate the text of the provided text using the CMP mechanism

        : param data: str the text to obfuscate
        : param numberOfCores: int the number of cores to use for the obfuscation

        : return: str the obfuscated text

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)
        >>> data: str = 'This is a query to obfuscate'
        >>> numberOfCores: int = 4
        >>> mech1.obfuscateText(data, numberOfCores)
        '''
        words: List[str] = data.split() #split query into words
        results: List = self.multiCoreObfuscateText(words, numberOfCores) #flatten the results
        return results
                                 
    def noisyEmb(self, words: List[str]) -> np.array: 
        '''
        method noisyEmb: this method is used to add noise to the embeddings of the words

        : param words: List[str] the list of words to add noise to
        : return: np.array the noisy embeddings

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)
        >>> words: List[str] = ['word1', 'word2', 'word3']
        >>> mech1.noisyEmb(words)
        '''
        embs: List[np.array] = []
        for word in words:
            if word not in self.vocab.embeddings:
                embs.append(
                    np.zeros(self.embMatrix.shape[1]) + npr.normal(0, 1, self.embMatrix.shape[1]) #handle OoV words
                    + self.pullNoise()
                    )
            else:
                embs.append(self.vocab.embeddings[word] + self.pullNoise())
        return np.array(embs)

    def processQuery(self, 
                     embs: np.array) -> str:
        '''
        method processQuery: this method is used to process the query and return the obfuscated query

        : param embs: np.array the embeddings of the words
        : return: str the obfuscated query

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)

        # Assuming that the embeddings of the words are known, e.g.: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> embs: np.array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> mech1.processQuery(embs)
        '''

        length: int = len(embs)
        distance: np.array = self.euclideanDistance(embs, self.embMatrix)
        closest: np.array = np.argpartition(distance, 1, axis=1)[:, :1]
        finalQuery: List[str] = []
        for i in range(length):
            finalQuery.append(list(self.vocab.embeddings.keys())[closest[i][0]])
        return ' '.join(finalQuery)
    
    @staticmethod
    def euclideanDistance(x: np.array, 
                          y: np.array) -> np.array:
        '''
        method euclideanDistance: this method is used to compute the euclidean distance between two matrices

        Remark: this method is an obtimization of the euclidean distance computation between two matrices

        : param x: np.array the first matrix
        : param y: np.array the second matrix
        : return: np.array the euclidean distance matrix between the two matrices

        Usage example:
        >>> x: np.array = np.array([1, 2, 3])
        >>> y: np.array = np.array([4, 5, 6])
        >>> euclideanDistance(x, y)
        '''
        x: np.array = np.array(x)
        y: np.array = np.array(y)
        x_expanded: np.array = x[:, np.newaxis, :]
        y_expanded: np.array = y[np.newaxis, :, :]
        return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))
    
    @staticmethod
    def multiCoreObfuscateText(self, words: List[str],
                               numberOfCores: int) -> List[str]:
        '''
        method multiCoreObfuscateText: this method is used to obfuscate the text parallelly

        : param words: List[str] the list of words to obfuscate
        : param numberOfCores: int the number of cores to use for the obfuscation

        : return: List[str] the list of obfuscated queries

        Usage example:

        >>> words: List[str] = ['word1', 'word2', 'word3']
        >>> numberOfCores: int = 4
        >>> multiCoreObfuscateText(words, numberOfCores)
        '''

        with mp.Pool(numberOfCores) as p: #use multiprocessing to speed up the obfuscation
            tasks = [self.noisyEmb(words) for i in range(numberOfCores)]
            results.append(p.map(self.processQuery, tasks))
        results = [item for sublist in results for item in sublist]
        return results