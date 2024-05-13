from ..AbstractTextObfuscationDPMechanism import AbstractTextObfuscationDPMechanism
import numpy as np
import pandas as pd
from typing import List
import numpy.random as npr

class AbstractEmbeddingPerturbationMechanism(AbstractTextObfuscationDPMechanism):
    '''
    Abstract class scramblingEmbeddingsMechanism: this class is used to define the abstract class of the scrambling Embeddings mechanisms
    '''
    def __init__(self, kwargs: dict[str:object]) -> None:
        '''
        Constructur of the scramblingEmbeddingsMechanism class
        '''
        super().__init__(kwargs)
    
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
        obfuscatedText = pd.concat([data['text'].apply(lambda x: self.processText(self.noisyEmb(x))) for _ in range(numberOfTimes)], axis=1)#note that here we're passing the embeddings noisy
        df = pd.concat([data, pd.DataFrame({'obfuscatedText': obfuscatedText.values.tolist()})], axis=1)
        df['text'] = df['text'].apply(lambda x: ' '.join(x))
        return df