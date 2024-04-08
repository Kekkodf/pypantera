from .utils.vocab import Vocab

import numpy as np
import os
from typing import List
import numpy.random as npr
import multiprocessing as mp

class Mechanism():
    '''
    Abstract class Mechanism: this class is used to define the abstract class of the mechanism
    '''
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
        assert 'epsilon' in kwargs, 'The epsilon parameter must be provided'
        assert kwargs['epsilon'] > 0, 'The epsilon parameter must be greater than 0'
        self.epsilon: float = kwargs['epsilon'] #set the epsilon parameter
    
    def noisyEmb(self, words: List[str]) -> np.array:
        '''
        method noisyEmb: this method is used to add noise to the embeddings of the words

        : param words: List[str] the list of words to add noise to
        : return: np.array the noisy embeddings

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)
        >>> words: List[str] = ['what', 'is', 'the', 'capitol', 'of', 'france']
        >>> mech1.noisyEmb(words)
        '''

        embs: List = []
        for word in words:
            if word not in self.vocab.embeddings:
                embs.append(
                    np.zeros(self.embMatrix.shape[1]) + npr.normal(0, 1, self.embMatrix.shape[1]) #handle OoV words
                    + self.pullNoise()
                    )
            else:
                embs.append(self.vocab.embeddings[word] + self.pullNoise())
        return np.array(embs)
    
    def obfuscateText(self, data: str, numberOfCores: int) -> List[str]:
        '''
        method obfuscateText: this method is used to obfuscate the text of the provided text 
        using the Mahalanobis mechanism

        : param data: str the text to obfuscate
        : param numberOfCores: int the number of cores to use for the obfuscation
        : return: str the obfuscated text

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)

        >>> text: str = 'what is the capitol of france'
        >>> mech1.obfuscateText(text, 1)
        '''

        words = data.split() #split query into words
        results: List = []
        with mp.Pool(numberOfCores) as p:
            tasks = [self.noisyEmb(words) for i in range(numberOfCores)]
            results.append(p.map(self.processQuery, tasks))
        results = [item for sublist in results for item in sublist]
        return results
    
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