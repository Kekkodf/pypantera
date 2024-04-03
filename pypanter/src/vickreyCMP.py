from .mechanism import Mechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List

class VickreyCMP(Mechanism):
    '''
    BibTeX of Vickrey (CMP based) Mechanism, extends Mahalanobis mechanism class of the pypanter package:

    @article{Xu2021OnAU,
    title={On a Utilitarian Approach to Privacy Preserving Text Generation},
    author={Zekun Xu and Abhinav Aggarwal and Oluwaseyi Feyisetan and Nathanael Teissier},
    journal={ArXiv},
    year={2021},
    volume={abs/2104.11838},
    url={https://www.semanticscholar.org/reader/dfd8fc9966ca8ec5c8bdc2dfc94099285f0e07a9}
    }
    '''
    def __init__(self, kwargs: dict) -> None:
        '''
        Initialization of the Mahalanobis Object

        : param kwargs: dict the dictionary containing the parameters of the Mechanism Object 
                        + the specific parameters of the Mahalanobis Mechanism (lambda)

        Once the Mahalanobis Object is created, the user can use the obfuscateText method to obfuscate 
        the text of the provided text.

        The attributes of the Mechanism Object are:
        - vocab: the vocabulary object containing the embeddings
        - embMatrix: the matrix containing the embeddings
        - index2word: the dictionary containing the index to word mapping
        - word2index: the dictionary containing the word to index mapping
        - epsilon: the epsilon parameter of the mechanism
        - lam: the lambda parameter of the Mahalanobis mechanism
        - sigma_loc: parameter used for pulling noise in obfuscation

        Usage example:
        >>> embPath: str = 'pathToMyEmbeddingsFile.txt'
        >>> eps: float = 0.1 #anyvalue of epsilon must be greater than 0
        >>> lam: float = 0.1 #anyvalue of lambda must be between 0 and 1
        >>> mech1 = Mahalanobis({'embPath': embPath, 'epsilon': eps, 'lambda': lam})
        '''
        super().__init__(kwargs)
        assert 't' in kwargs, 'The t parameter must be provided'
        assert kwargs['t'] >= 0 and kwargs['t'] <= 1, 'The t parameter must be between 0 and 1'
        self.t: float = kwargs['t']

    def processQuery(self, 
                     embs: np.array) -> str:
        length: int = len(embs) #compute number of words
        distance: np.array = self.euclideanDistance(embs, self.embMatrix) #compute distances between words and embeddings in the vocabulary
        closest: np.array = np.argpartition(distance, 2, axis=1)[:, :2] #find the two closest embeddings for each word
        distToClosest: np.array = distance[np.tile(np.arange(length).reshape(-1,1),2), closest] #compute the distances to the two closest embeddings
        p = ((1- self.t) * distToClosest[:,1]) / (self.t * distToClosest[:,0] + (1 - self.t) * distToClosest[:, 1]) #compute the probabilities of choosing the second closest embedding
        vickreyChoise: np.array = np.array([npr.choise(2, p=[p[w], 1-p[w]]) for w in range(length)]) #choose the closest embedding according to the probabilities
        noisyEmbeddings: np.array = self.embMatrix[closest[np.arange(length), vickreyChoise]] #get the noisy embeddings
        finalQuery: List[str] = []
        for i in range(length):
            finalQuery.append(list(self.vocab.embeddings.keys())[noisyEmbeddings[i][0]]) #get the words corresponding to the noisy embeddings
        return ' '.join(finalQuery)