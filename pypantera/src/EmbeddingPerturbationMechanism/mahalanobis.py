from .AbstractEmbeddingPerturbationMechanism import AbstractEmbeddingPerturbationMechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
from typing import List

'''
# Mahalanobis Mechanism
'''

class Mahalanobis(AbstractEmbeddingPerturbationMechanism):
    '''
    # Mahalanobis
    
    BibTeX of Mahalanobis Mechanism, extends Mechanism mechanism class of the pypantera package:

    @inproceedings{xu-etal-2020-differentially,
    title = "A Differentially Private Text Perturbation Method Using Regularized Mahalanobis Metric",
    author = "Xu, Zekun and Aggarwal, Abhinav and Feyisetan, Oluwaseyi and Teissier, Nathanael",
    editor = "Feyisetan, Oluwaseyi and Ghanavati, Sepideh  and Malmasi, Shervin and Thaine, Patricia",
    booktitle = "Proceedings of the Second Workshop on Privacy in NLP",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.privatenlp-1.2.pdf",
    doi = "10.18653/v1/2020.privatenlp-1.2",
    pages = "7--17",
    abstract = "Balancing the privacy-utility tradeoff is a crucial requirement of many practical machine learning systems that deal with sensitive customer data. A popular approach for privacy- preserving text analysis is noise injection, in which text data is first mapped into a continuous embedding space, perturbed by sampling a spherical noise from an appropriate distribution, and then projected back to the discrete vocabulary space. While this allows the perturbation to admit the required metric differential privacy, often the utility of downstream tasks modeled on this perturbed data is low because the spherical noise does not account for the variability in the density around different words in the embedding space. In particular, words in a sparse region are likely unchanged even when the noise scale is large. In this paper, we propose a text perturbation mechanism based on a carefully designed regularized variant of the Mahalanobis metric to overcome this problem. For any given noise scale, this metric adds an elliptical noise to account for the covariance structure in the embedding space. This heterogeneity in the noise scale along different directions helps ensure that the words in the sparse region have sufficient likelihood of replacement without sacrificing the overall utility. We provide a text-perturbation algorithm based on this metric and formally prove its privacy guarantees. Additionally, we empirically show that our mechanism improves the privacy statistics to achieve the same level of utility as compared to the state-of-the-art Laplace mechanism.",
    }
    '''
    def __init__(self, kwargs: dict[str:object]) -> None:

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
        assert 'lam' in kwargs, 'The lambda parameter must be provided'
        assert kwargs['lam'] >= 0 and kwargs['lam'] <= 1, 'The lambda parameter must be between 0 and 1'
        self.lam: float = kwargs['lam']
        cov_mat = np.cov(self.embMatrix.T, ddof=0)
        sigma = cov_mat/ np.mean(np.var(self.embMatrix.T, axis=1))
        self._sigmaLoc = sqrtm(self.lam * sigma + (1 - self.lam) * np.eye(self.embMatrix.shape[1]))   
        self.name:str = 'Mahalanobis'   

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

        N: np.array = npr.multivariate_normal(
            np.zeros(self.embMatrix.shape[1]), 
            np.eye(self.embMatrix.shape[1])
            )
        X: np.array = N / np.sqrt(np.sum(N ** 2))
        X: np.array = np.dot(self._sigmaLoc, X)
        X: np.array = X / np.sqrt(np.sum(X ** 2))
        Y: np.array = npr.gamma(
            self.embMatrix.shape[1], 
            1 / self.epsilon
            )
        Z: np.array = Y * X
        return Z

    def processText(self, 
                     embs: np.array) -> str:
        
        '''
        method processText: this method is used to process the Text and return the obfuscated Text

        : param embs: np.array the embeddings of the words
        : return: str the obfuscated Text

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)

        # Assuming that the embeddings of the words are known, e.g.: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> embs: np.array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> mech1.processText(embs)
        '''
        
        length: int = len(embs)
        distance: np.array = self.euclideanDistance(embs, self.embMatrix)
        closest: np.array = np.argpartition(distance, 1, axis=1)[:, :1]
        finalText: List[str] = []
        for i in range(length):
            finalText.append(list(self.vocab.embeddings.keys())[closest[i][0]])
        return ' '.join(finalText)