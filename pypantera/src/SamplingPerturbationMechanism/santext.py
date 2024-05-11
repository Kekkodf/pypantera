from .AbstractSamplingPerturbationMechanism import AbstractSamplingPerturbationMechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List
import time

'''
# SanText mechanism
'''

class SanText(AbstractSamplingPerturbationMechanism):
    '''
    # SanText Mechansim
    
    BibTeX of SanText Mechanism, extends CMP mechanism class of the pypanter package:

    @inproceedings{ACL21/YueDu21,
    author    = {Xiang Yue and Minxin Du and Tianhao Wang and Yaliang Li and Huan Sun and Sherman S. M. Chow},
    title     = {Differential Privacy for Text Analytics via Natural Text Sanitization},
    booktitle = {Findings, {ACL-IJCNLP} 2021},
    year      = {2021},
    }
    '''
    def __init__(self, kwargs: dict[str:object]) -> None:
        '''
        Initialization of the SanText Object

        : param kwargs: dict the dictionary containing the parameters of the Mechanism Object 
                        + the specific parameters of the SanText Mechanism ()

        Once the SanText Object is created, the user can use the obfuscateText method to obfuscate 
        the text of the provided text.

        The attributes of the Mechanism Object are:
        - vocab: the vocabulary object containing the embeddings
        - embMatrix: the matrix containing the embeddings
        - index2word: the dictionary containing the index to word mapping
        - word2index: the dictionary containing the word to index mapping
        - epsilon: the epsilon parameter of the mechanism
        - 

        Usage example:
        >>> embPath: str = 'pathToMyEmbeddingsFile.txt'
        >>> eps: float = 0.1 #anyvalue of epsilon must be greater than 0
        >>> lam: float = 0.1 #anyvalue of lambda must be between 0 and 1
        >>> mech1 = SanText({'embPath': embPath, 'epsilon': eps, })
        '''
        super().__init__(kwargs)
        self.name:str = 'SanText'
        self.prob_matrix = {}

    def processText(self, text:List[str])->str:
        '''
        method selfProcessQueryText: this method is used to process the Text and return the obfuscated Text

        : param embs: np.array the embeddings of the words
        : return: str the obfuscated Text
        '''
        length:int = len(text) #number of words in the text
        text_tuple:tuple = tuple(text)
        #get embeddings
        if (text_tuple, self.epsilon) not in self.prob_matrix.keys():
            embs = []
            for word in text:
                if word not in self.vocab.embeddings.keys():
                    embs.append(np.zeros(self.embMatrix.shape[1]) + npr.normal(0, 1, self.embMatrix.shape[1]))
                else:
                    embs.append(self.vocab.embeddings[word])
            embs:np.array = np.array(embs)
        # Compute distance matrix between word and all words in vocabulary
            distance:np.array = self.euclideanDistance(embs, self.embMatrix)
            # Compute probability
            exp_neg_distance:np.array = np.exp(-0.5 * self.epsilon * distance)
            total_sum:np.array = np.reciprocal(np.sum(exp_neg_distance, axis=1))
            probabilities:np.array = exp_neg_distance * total_sum[:, np.newaxis]
            
            # Normalize
            probabilities /= np.sum(probabilities, axis=1, keepdims=True)
            
            # Add the probabilities to the dictionary
            self.prob_matrix[(text_tuple, self.epsilon)] = probabilities
        else:
            probabilities = self.prob_matrix[(text_tuple, self.epsilon)]
        
        # Sampled obfuscated word should be a list of length len(text)
        sampled_indices:List[np.array] = [np.random.choice(self.embMatrix.shape[0], p=prob) for prob in probabilities]
        sampled_obfuscated_text:List[str] = [list(self.vocab.embeddings.keys())[index] for index in sampled_indices]
        
        return " ".join(sampled_obfuscated_text)
        
        