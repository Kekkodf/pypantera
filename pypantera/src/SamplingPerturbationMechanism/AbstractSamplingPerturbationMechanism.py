from ..AbstractTextObfuscationDPMechanism import AbstractTextObfuscationDPMechanism
import pandas as pd
from typing import List

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
        obfuscatedText = pd.concat([data['text'].apply(lambda x: self.processText(x)) for _ in range(numberOfTimes)], axis=1) #nopte that here we're passing the list of words
        df = pd.concat([data, pd.DataFrame({'obfuscatedText': obfuscatedText.values.tolist()})], axis=1)
        df['text'] = df['text'].apply(lambda x: ' '.join(x))
        return df