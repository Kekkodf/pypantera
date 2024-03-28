import numpy as np
import numpy.random as npr
import multiprocessing as mp
from .utils.vocab import Vocab
import os
from typing import List

'''
    BibTeX of CMP Mechanism, base mechanism class of the pypanter package:

    @inproceedings{FeyisetanEtAl2020CMP,
    author = {Feyisetan, Oluwaseyi and Balle, Borja and Drake, Thomas and Diethe, Tom},
    title = {Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations},
    year = {2020},
    isbn = {9781450368223},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3336191.3371856},
    doi = {10.1145/3336191.3371856},
    abstract = {Accurately learning from user data while providing quantifiable privacy guarantees provides an opportunity to build better ML models while maintaining user trust. This paper presents a formal approach to carrying out privacy preserving text perturbation using the notion of d_χ-privacy designed to achieve geo-indistinguishability in location data. Our approach applies carefully calibrated noise to vector representation of words in a high dimension space as defined by word embedding models. We present a privacy proof that satisfies d_χ-privacy where the privacy parameter $varepsilon$ provides guarantees with respect to a distance metric defined by the word embedding space. We demonstrate how $varepsilon$ can be selected by analyzing plausible deniability statistics backed up by large scale analysis on GloVe and fastText embeddings. We conduct privacy audit experiments against $2$ baseline models and utility experiments on 3 datasets to demonstrate the tradeoff between privacy and utility for varying values of varepsilon on different task types. Our results demonstrate practical utility (< 2\% utility loss for training binary classifiers) while providing better privacy guarantees than baseline models.},
    booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining},
    pages = {178-186},
    numpages = {9},
    keywords = {privacy, plausible deniability, differential privacy},
    location = {Houston, TX, USA},
    series = {WSDM '20}
    }
'''

class Mechanism():
    '''
    Class Mechanism: this class is used to create a mechanism object that obfuscate a provided Query Object
    '''
    
    def __init__(self, kwargs: dict) -> None:

        '''
        Initialization of the Mechanism Object

        : param kwargs: dict the dictionary containing the parameters of the Mechanism Object

        Once the Mechanism Object is created, the user can use the obfuscateText method to obfuscate 
        the text of the provided text.

        The attributes of the Mechanism Object are:
        - vocab: the vocabulary object containing the embeddings
        - embMatrix: the matrix containing the embeddings
        - index2word: the dictionary containing the index to word mapping
        - word2index: the dictionary containing the word to index mapping
        - epsilon: the epsilon parameter of the mechanism

        Usage example:
        >>> embPath: str = 'pathToMyEmbeddingsFile.txt'
        >>> eps: float = 0.1 #anyvalue of epsilon must be greater than 0
        >>> mech1 = Mechanism({'embPath': embPath, 'epsilon': eps})
        '''
        self.vocab: Vocab = Vocab(kwargs['embPath'])
        self.embMatrix: np.array = np.array(
            list(self.vocab.embeddings.values())
            )
        self.index2word: dict = {
            i: word 
            for i, word in enumerate(self.vocab.embeddings.keys())
            }
        self.word2index: dict = {
            word: i 
            for i, word in enumerate(self.vocab.embeddings.keys())
            }

        assert 'epsilon' in kwargs, 'The epsilon parameter must be provided'
        assert kwargs['epsilon'] > 0, 'The epsilon parameter must be greater than 0'
        self.epsilon: float = kwargs['epsilon']

    def obfuscateText(self, data: str, numberOfCores: int) -> List[str]:
        '''
        method obfuscateText: this method is used to obfuscate the text of the provided text using the CMP mechanism

        : param data: str the text to obfuscate
        : param numberOfCores: int the number of cores to use for the obfuscation

        : return: str the obfuscated text
        '''
        words = data.split() #split query into words
        results = []
        with mp.Pool(numberOfCores) as p:
            tasks = [self.noisyEmb(words) for i in range(numberOfCores)]
            results.append(p.map(self.processQuery, tasks))
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

    def pullNoise(self) -> np.array:
        '''
        method pullNoise: this method is used to pull noise accordingly 
        to the definition of the CMP mechanism, see BibTeX ref

        : return: np.array the noise pulled

        Usage example:
        (Considering that the Mechanism Object mech1 has been created 
        as in the example of the __init__ method)
        >>> mech1.pullNoise()
        '''
        N = self.epsilon * npr.multivariate_normal(
            np.zeros(self.embMatrix.shape[1]),
            np.eye(self.embMatrix.shape[1]))
        X = N / np.sqrt(np.sum(N ** 2))
        Y = npr.gamma(
            self.embMatrix.shape[1],
            1 / self.epsilon)
        Z = Y * X
        return Z
    
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
        x = np.array(x)
        y = np.array(y)
        x_expanded = x[:, np.newaxis, :]
        y_expanded = y[np.newaxis, :, :]
        return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))