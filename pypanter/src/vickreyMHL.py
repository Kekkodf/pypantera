from .mahalanobis import Mahalanobis
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List

class VickreyMhl(Mahalanobis):
    '''
    BibTeX of Vickrey (Mahalanobis based) Mechanism, extends Mahalanobis mechanism class of the pypanter package:

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
        assert 'lambda' in kwargs, 'The lambda parameter must be provided'
        assert kwargs['lambda'] >= 0 and kwargs['lambda'] <= 1, 'The lambda parameter must be between 0 and 1'
        self.lam: float = kwargs['lambda']
        cov_mat = np.cov(self.embMatrix.T, ddof=0)
        sigma = cov_mat/ np.mean(np.var(self.embMatrix.T, axis=1))
        self.sigmaLoc = sqrtm(self.lam * sigma + (1 - self.lam) * np.eye(self.embMatrix.shape[1]))      

    def pullNoise(self) -> np.array:
        '''
        method pullNoise: this method is used to pull noise accordingly 
        to the definition of the Mahalanobis mechanism, see BibTeX ref

        : return: np.array the noise pulled

        Usage example:
        (Considering that the Mechanism Object mech1 has been created 
        as in the example of the __init__ method)
        >>> mech1.pullNoise()
        '''
        N = npr.multivariate_normal(
            np.zeros(self.embMatrix.shape[1]), 
            np.eye(self.embMatrix.shape[1])
            )
        X = N / np.sqrt(np.sum(N ** 2))
        X = np.dot(self.sigmaLoc, X)
        X = X / np.sqrt(np.sum(X ** 2))
        Y = npr.gamma(
            self.embMatrix.shape[1], 
            1 / self.epsilon
            )
        Z = Y * X
        return Z
    
    def obfuscateText(self, data: str, numberOfCores: int) -> List[str]:
        '''
        method obfuscateText: this method is used to obfuscate the text of the provided text 
        using the Mahalanobis mechanism

        : param data: str the text to obfuscate
        : param numberOfCores: int the number of cores to use for the obfuscation

        : return: str the obfuscated text
        '''
        words = data.split() #split query into words
        results = []
        with mp.Pool(numberOfCores) as p:
            tasks = [self.noisyEmb(words) for i in range(numberOfCores)]
            results.append(p.map(self.processQuery, tasks))
        results = [item for sublist in results for item in sublist]
        return results

    def noisyEmb(self, words: List[str]) -> np.array:
        embs = []
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
        length = len(embs)
        distance = self.euclideanDistance(embs, self.embMatrix)
        closest = np.argpartition(distance, 1, axis=1)[:, :1]
        finalQuery = []
        for i in range(length):
            finalQuery.append(list(self.vocab.embeddings.keys())[closest[i][0]])
        return ' '.join(finalQuery)