from .mechanism import Mechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List

class Mahalanobis(Mechanism):
    '''
    BibTeX of Mahalanobis Mechanism, extends CMP mechanism class of the pypanter package:
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