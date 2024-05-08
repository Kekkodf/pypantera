from .mechanism import Mechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List
import time

class TEM(Mechanism):
    '''
    BibTeX of TEM Mechanism, extends CMP mechanism class of the pypanter package:

    @article{DBLP:journals/corr/abs-2107-07928,
    author       = {Ricardo Silva Carvalho and Theodore Vasiloudis and Oluwaseyi Feyisetan},
    title        = {{TEM:} High Utility Metric Differential Privacy on Text},
    journal      = {CoRR},
    volume       = {abs/2107.07928},
    year         = {2021},
    url          = {https://arxiv.org/abs/2107.07928},
    eprinttype    = {arXiv},
    eprint       = {2107.07928},
    timestamp    = {Wed, 21 Jul 2021 15:55:35 +0200},
    biburl       = {https://dblp.org/rec/journals/corr/abs-2107-07928.bib},
    bibsource    = {dblp computer science bibliography, https://dblp.org}
    }
    '''
    def __init__(self, kwargs: dict[str:object]) -> None:
        '''
        Initialization of the TEM Object

        : param kwargs: dict the dictionary containing the parameters of the Mechanism Object 
                        + the specific parameters of the TEM Mechanism ()

        Once the TEM Object is created, the user can use the obfuscateText method to obfuscate 
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
        >>> beta: float = 0.001 #anyvalue of beta must be greater than 0 and less than 1
        >>> mech1 = TEM({'embPath': embPath, 'epsilon': eps, })
        '''
        super().__init__(kwargs)
        assert 'beta' in kwargs, 'The beta parameter must be provided'
        assert kwargs['beta'] > 0 and kwargs['beta'] < 1, 'The beta parameter must be between 0 and 1'
        self.beta: float = kwargs['beta'] 
        self.gamma: float = (2/self.epsilon)*np.log(((1-self.beta)*(len(self.embMatrix)-1))/self.beta)

        self.candidates: dict = {}
        #self._internalPreprocessing()

    def _internalPreprocessing(self) -> List[dict]:
        '''
        private method _internalPreprocessing: this method is used to preprocess the embeddings matrix accordingly to TEM mechanism
        '''
        #get the list of words
        words: List[str] = list(self.vocab.embeddings.keys())
        #compute the distances between the embeddings in the vocabulary
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(self.speedUp, [(words, w) for w in words])

    def speedUp(self, t:tuple) -> None:
        words:list = t[0]
        w:str = t[1]
        distances = self.euclideanDistance(np.array(self.vocab.embeddings[w]).reshape(1,-1), np.array(self.embMatrix))
        distances = distances[0]
        #create L_w selecting the words that are below the gamma
        Lw = [(words[i], distances[i]) for i in range(len(words)) if distances[i] < self.gamma]
        #create L_hat_w selecting the words that are in self.vocab.embeddings.keys() not in Lw
        L_hat_w = [(words[i], distances[i]) for i in range(len(words)) if distances[i] >= self.gamma]
        try:
            score = -self.gamma+2*np.log(len(L_hat_w)/self.epsilon)
            Lw.append(('PLACE_HOLDER_ITEM', score))
        except RuntimeWarning:
            score = -np.inf
            Lw.append(('PLACE_HOLDER_ITEM', score))
        self.candidates.update({w: (Lw, L_hat_w)})


    def pullNoise(self) -> np.array:
        '''
        pullNoise method: this method is used to pull noise from the Laplace distribution
        : param n: int the number of noise to pull (size of the Lw precomputed list)

        : return: np.array the noise pulled from the Laplace distribution

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)
        >>> mech1.pullNoise()
        '''
        gumbel_mean: float = 0
        gumbel_scale: float = 2/self.epsilon
        return npr.gumbel(gumbel_mean, gumbel_scale, self.embMatrix.shape[1])

    def processQuery(self, 
                 embs: np.array) -> str:
        '''
        processQuery method: this method is used to process the query accordingly
        to the definition of the TEM mechanism, see BibTeX ref

        : param embs: np.array the embeddings of the query
        : return: str the obfuscated query

        Usage example:
        >>> embPath: str = 'pathToMyEmbeddingsFile.txt'
        >>> eps: float = 0.1
        >>> beta: float = 0.001
        >>> mech1 = TEM({'embPath': embPath, 'epsilon': eps, 'beta': beta})
        >>> words: List[str] = ['what is the capitol of france']
        >>> embs: np.array = mech1.getEmbeddings(words)
        >>> obfuscatedQuery: str = mech1.processQuery(embs)
        '''
        #check the value of the embeddings returning the keys of the embeddings
        words: List[str] = [self.vocab.embeddings.keys()[np.where(self.vocab.embeddings.values() == emb)] for emb in embs]
        




        




        ...