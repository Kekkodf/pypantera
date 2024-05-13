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

    It extends the AbstractTextObfuscationDPMechanism class
    '''
    def __init__(self, kwargs: dict[str:object]) -> None:
        '''
        Constructur of the scramblingEmbeddingsMechanism class
        '''
        super().__init__(kwargs)
    
    def obfuscateText(self, data:pd.DataFrame, numberOfTimes: int) -> List[str]:
        '''
        # Obfuscate Text

        method obfuscateText: this method is used to obfuscate the text of the provided text 
        using the Mahalanobis mechanism

        It takes a Pandas DataFrame as input and returns a Pandas DataFrame as output

        The input should have the following columns:
        - id: the id of the text
        - text: the text to obfuscateù

        The output will have the following columns:
        - id: the id of the text
        - text: the original text
        - obfuscatedText: the obfuscated text

        : param data: pd.DataFrame with the text to obfuscate
        : param numberOfTimes: int the number of times to obfuscate the text
        : return: pd.DataFrame with the original texts and the obfuscated ones

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)

        >>> text: pd.DataFrame = pd.DataFrame({'id': [1], 'text': ['what is the capitol of france']})
        >>> mech1.obfuscateText(text, 1)
        '''
        #rename the columns
        data.columns = ['id', 'text']
        #tokenize the text column
        data['text'] = data['text'].apply(lambda x: x.lower().split())
        df:pd.DataFrame = pd.DataFrame(columns=['id', 'text', 'obfuscatedText'])
        #obfuscate the text column
        obfuscatedText:pd.DataFrame = pd.concat([data['text'].apply(lambda x: self.processText(x)) for _ in range(numberOfTimes)], axis=1) #nopte that here we're passing the list of words
        df = pd.concat([data, pd.DataFrame({'obfuscatedText': obfuscatedText.values.tolist()})], axis=1)
        df['text'] = df['text'].apply(lambda x: ' '.join(x))
        return df