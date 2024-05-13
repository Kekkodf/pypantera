from .utils.vocab import Vocab
import numpy as np
import pandas as pd
from typing import List

class AbstractTextObfuscationDPMechanism():
    '''
    Abstract class Mechanism: this class is used to define mechanism
    '''
    def __init__(self, kwargs: dict[str:object]) -> None:
        self.vocab: Vocab = Vocab(kwargs['embPath']) #load the vocabulary
        self.embMatrix: np.array = np.array(
            list(self.vocab.embeddings.values())
            ) #load the embeddings matrix
        self._word2index: dict = {
            word: i 
            for i, word in enumerate(self.vocab.embeddings.keys())
            } 
        self._index2word: dict = {
            i: word for word,i in self._word2index.items() 
        }        
        assert 'epsilon' in kwargs, 'The epsilon parameter must be provided'
        assert kwargs['epsilon'] > 0, 'The epsilon parameter must be greater than 0'
        self.epsilon: float = kwargs['epsilon'] #set the epsilon parameter
    
    def indexes2words(self, indexes:list) -> List[str]:
        return [self._index2word[e] for f in indexes for e in f]
    
    def obfuscateText(self, 
                      data:pd.DataFrame, 
                      numberOfTimes: int) -> List[str]:
        '''
        method obfuscateText: this method is used to obfuscate the text of the provided text 
        it uses the istance specific obfuscationText method
        

        : param data: pd.DataFrame of the texts to obfuscate
        : param numberOfTimes: int the number of times to obfuscate the text
        : return: df the obfuscated texts

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)

        >>> text: str = 'what is the capitol of france'
        >>> mech1.obfuscateText(text, 1)
        '''
        return self.obfuscateText(data, numberOfTimes)

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
    def cosineSimilarity(x: np.array,
                         y: np.array) -> np.array:
        '''
        method cosineSimilarity: this method is used to compute the cosine similarity between two matrices

        Remark: this method is an obtimization of the cosine similarity computation between two matrices

        : param x: np.array the first matrix
        : param y: np.array the second matrix
        : return: np.array the cosine similarity matrix between the two matrices

        Usage example:
        >>> x: np.array = np.array([1, 2, 3])
        >>> y: np.array = np.array([4, 5, 6])
        >>> cosineSimilarity(x, y)
        '''
        x: np.array = np.array(x)
        y: np.array = np.array(y)
        x_expanded: np.array = x[:, np.newaxis, :]
        y_expanded: np.array = y[np.newaxis, :, :]
        return np.sum(x_expanded * y_expanded, axis=2) / (np.linalg.norm(x_expanded, axis=2) * np.linalg.norm(y_expanded, axis=2))