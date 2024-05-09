from ..AbstractTextObfuscationDPMechanism import AbstractTextObfuscationDPMechanism

import numpy as np
import pandas as pd
import os
from typing import List
import numpy.random as npr
import multiprocessing as mp

class AbstractSamplingPerturbationMechanism(AbstractTextObfuscationDPMechanism):
    '''
    Abstract class scramblingSamplingMechanism: this class is used to define the abstract class of the scrambling Sampling mechanisms
    '''
    def __init__(self, kwargs: dict[str:object]) -> None:
        '''
        Constructur of the scramblingEmbeddingsMechanism class
        '''
        super().__init__(kwargs)
    
    def obfuscateText(self, data:pd.DataFrame, numberOfTimes: int) -> List[str]:
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

        
        #rename the columns
        data.columns = ['id', 'text']
        #tokenize the text column
        data['text'] = data['text'].apply(lambda x: x.lower().split())
        df = pd.DataFrame(columns=['id', 'text', 'obfuscatedText'])
        #obfuscate the text column
        obfuscatedText = pd.concat([data['text'].apply(lambda x: self.processQuery(x)) for _ in range(numberOfTimes)], axis=1)
        #df = pd.concat([data, pd.DataFrame({'obfuscatedText': obfuscatedText.values.tolist()})], axis=1)
        #df['text'] = df['text'].apply(lambda x: ' '.join(x))
        #return df
        ...
    
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