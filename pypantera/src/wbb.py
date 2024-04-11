from .mechanism import Mechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List
import nltk
from nltk import pos_tag

class WBB(Mechanism):
    '''
    BibTeX of WBB Mechanism, extends CMP mechanism class of the pypanter package:

    TBP (To Be Published)
    
    '''
    def __init__(self, kwargs: dict[str:object]) -> None:
        '''
        Initialization of the WBB Object

        : param kwargs: dict the dictionary containing the parameters of the Mechanism Object 
                        + the specific parameters of the WBB Mechanism ()

        Once the WBB Object is created, the user can use the obfuscateText method to obfuscate 
        the text of the provided text.

        The attributes of the Mechanism Object are:
        - vocab: the vocabulary object containing the embeddings
        - embMatrix: the matrix containing the embeddings
        - index2word: the dictionary containing the index to word mapping
        - word2index: the dictionary containing the word to index mapping
        - epsilon: the epsilon parameter of the mechanism
        - n: the n parameter of the WBB mechanism
        - k: the k parameter of the WBB mechanism
        - posTags: the list of POS tags to consider

        Usage example:
        >>> embPath: str = 'pathToMyEmbeddingsFile.txt'
        >>> eps: float = 0.1 #anyvalue of epsilon must be greater than 0
        >>> lam: float = 0.1 #anyvalue of lambda must be between 0 and 1
        >>> mech1 = WBB({'embPath': embPath, 'epsilon': eps, })
        '''
        super().__init__(kwargs)
        assert 'n' in kwargs, 'The n parameter must be provided and greater than 0'
        assert kwargs['n'] > 0, 'The n parameter must be greater than 0'
        self.n: int = kwargs['n']
        assert 'k' in kwargs, 'The k parameter must be provided and greater than 0'
        assert kwargs['k'] > 0, 'The k parameter must be greater than 0'
        self.k: int = kwargs['k']
        assert 'listOfTags' in kwargs, 'The listOfTags parameter must be provided'
        self.posTags: List[str] = kwargs['listOfTags']
        assert 'metricFunction' in kwargs, 'The metricFunction parameter must be provided'
        self.metricFunction: str = kwargs['metricFunction']
        

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
        self.wordsNotConsidered: List[str] = []
        embs: List = []
        #compute the taggs of the words
        words: List[tuple] = pos_tag(words)
        for word in words:
            if word[1] in self.posTags:
                if word[0] not in self.vocab.embeddings:
                    embs.append(
                        np.zeros(self.embMatrix.shape[1]) + npr.normal(0, 1, self.embMatrix.shape[1]) #handle OoV words
                        )
                    self.wordsNotConsidered.append('_')
                else:
                    embs.append(self.vocab.embeddings[word[0]])
                    self.wordsNotConsidered.append('_')
            else:
                self.wordsNotConsidered.append(word[0])
        return np.array(embs)

    def processQuery(self,
                     embs: np.array) -> str:
        '''
        method processQuery: this method is used to process the query and return the obfuscated query

        : param embs: np.array the embeddings of the words
        : return: str the obfuscated query

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)

        # Assuming that the embeddings of the words are known, e.g.: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> embs: np.array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> mech1.processQuery(embs)
        '''
        candidatesList, distanceValues = self.mappingFunction(embs)
        sampledCandidates = self.samplingFunction(candidatesList, distanceValues)
        finalQuery = []

        #check if there are words that were considered
        for i in range(len(self.wordsNotConsidered)):
            if self.wordsNotConsidered[i] == '_':
                finalQuery.append(sampledCandidates.pop(0))
            else:
                finalQuery.append(self.wordsNotConsidered[i])
        print(self.wordsNotConsidered)
        
        return ' '.join(finalQuery)


    def mappingFunction(self, embs: np.array) -> List[str]:
        '''
        
        '''
        distance = self.distance(embs, self.embMatrix)
        candidates = np.argsort(distance, axis=1)[:, self.k:self.k+self.n]
        distanceValues = distance[np.arange(distance.shape[0])[:, None], candidates]

        candidatesList = [[self.index2word[c] for c in candidate] for candidate in candidates]
        return candidatesList, distanceValues
    
    def samplingFunction(self, candidatesList: List[str], distanceValues: np.array) -> List[str]:
        '''
        
        '''
        #compute the scores of the candidates
        mean = np.mean(distanceValues, axis=1)
        std = np.std(distanceValues, axis=1)
        scores = (distanceValues - mean[:, None]) / std[:, None]
        scores = np.exp(scores) / np.sum(np.exp(scores), axis=1)[:, None]

        #sample the candidates
        sampledCandidates = []
        for i in range(len(candidatesList)):
            sampledCandidates.append(npr.choice(candidatesList[i], p=scores[i]))
        return sampledCandidates
    
    def distance(self, x: np.array, y: np.array) -> np.array:
        '''
        
        '''
        if self.metricFunction == 'euclidean':
            return self.euclideanDistance(x, y)
        elif self.metricFunction == 'cosine':
            return self.cosineDistance(x, y)
        elif self.metricFunction == 'product':
            return self.productDistance(x, y)
    
    @staticmethod
    def cosineDistance(x: np.array, 
                          y: np.array) -> np.array:
        '''
        method cosineDistance: this method is used to compute the cosine distance between two matrices

        Remark: this method is an obtimization of the cosine distance computation between two matrices

        : param x: np.array the first matrix
        : param y: np.array the second matrix
        : return: np.array the cosine distance matrix between the two matrices

        Usage example:
        >>> x: np.array = np.array([1, 2, 3])
        >>> y: np.array = np.array([4, 5, 6])
        >>> cosineDistance(x, y)
        '''

        x: np.array = np.array(x)
        y: np.array = np.array(y)
        x_expanded: np.array = x[:, np.newaxis, :]
        y_expanded: np.array = y[np.newaxis, :, :]
        return 1 - np.sum(x_expanded * y_expanded, axis=2) / (np.linalg.norm(x_expanded, axis=2) * np.linalg.norm(y_expanded, axis=2))
    
    @staticmethod
    def productDistance(self,
                        x: np.array,
                        y: np.array) -> np.array:
        '''
        productDistance: this method is used to compute the product distance between two matrices
        calling the cosineDistance and euclideanDistance methods
        '''
        return self.cosineDistance(x, y) * self.euclideanDistance(x, y)
    