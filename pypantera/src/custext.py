from .mechanism import Mechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List

class CusText(Mechanism):
    '''
    BibTeX of CusText Mechanism, extends CMP mechanism class of the pypanter package:

    @inproceedings{ChenEtAl2023Customized,
    title = {A Customized Text Sanitization Mechanism with Differential Privacy},
    author = {Chen, Sai and Mo, Fengran and Wang, Yanhao and Chen, Cen and Nie, Jian-Yun and Wang, Chengyu and Cui, Jamie},
    editor = {Rogers, Anna and Boyd-Graber, Jordan and Okazaki, Naoaki},
    booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
    month = jul,
    year = {2023},
    address = {Toronto, Canada},
    publisher = {Association for Computational Linguistics},
    url = {https://aclanthology.org/2023.findings-acl.355},
    doi = {10.18653/v1/2023.findings-acl.355},
    pages = {5747--5758},
    }    
    '''
    def __init__(self, kwargs: dict[str:object]) -> None:
        '''
        Initialization of the CusText Object

        : param kwargs: dict the dictionary containing the parameters of the Mechanism Object 
                        + the specific parameters of the CusText Mechanism ()

        Once the CusText Object is created, the user can use the obfuscateText method to obfuscate 
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
        >>> mech1 = CusText({'embPath': embPath, 'epsilon': eps, })
        '''
        super().__init__(kwargs)
        assert 'k' in kwargs, 'The parameter k must be provided'
        assert kwargs['k'] > 0, 'The parameter k must be greater than 0'
        self.k = kwargs['k']

    def mappingFunction(self):
        '''
        mappingFunction of the CusText Mechanism

        : param text: List[str] the list of words of the text to be obfuscated

        : return: List[str] the list of tokens to map the words of the text

        Usage example:
        >>> text = ['I', 'love', 'privacy']
        >>> mech1.tokenMappingGeneration(text)
        '''
        X:List[str] = list(self.vocab.embeddings.keys())

        Y:List[str] = X

        f_map:dict[str:dict[str:np.array]] = {}

        Y_prime:List[str] = []
        while len(X) < self.k:
            #select a random word x from X
            x:str = X.pop(npr.randint(0, len(X)))
            #add x to Y_prime
            Y_prime.append(x)
            #get embeddings vectors for all y in Y
            y_embs:np.array = np.array([self.vocab.embeddings[y] for y in Y])
            #get the embeddings vectors for x
            x_emb:np.array = self.vocab.embeddings[x]
            #compute the euclidean distance between the embeddings of x and y
            distances:np.array = self.euclideanDistance(x_emb, y_embs)
            #sort the distances
            sorted_distances:np.array = np.argsort(distances)
            #select the k closest words to x
            k_closest = sorted_distances[:self.k]
            #define a dict where the key is the word x and the value is a dict with keys the words and values the distances
            f_map[x] = {Y[i]: distances[i] for i in k_closest}
            #update X as X\Y_prime, Y as Y\Y_prime and self.k as self.k = len(X)
            X = [word for word in X if word not in Y_prime]
            Y = [word for word in Y if word not in Y_prime]
            self.k = len(X)
        return f_map
    
    def samplingFunction():
        raise NotImplementedError
    
    def obfuscateText(self, data: str, numberOfCores: int) -> List[str]:
        raise NotImplementedError

            

