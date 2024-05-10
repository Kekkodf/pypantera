from .AbstractSamplingPerturbationMechanism import AbstractSamplingPerturbationMechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List
import time
from collections import defaultdict

'''
# CusText Mechanism
'''

class CusText(AbstractSamplingPerturbationMechanism):
    '''
    # CusText

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
        >>> k: int = 5 #any integer greater than 0 
        >>> mech1 = CusText({'embPath': embPath, 'epsilon': eps, })
        '''
        super().__init__(kwargs)
        assert 'k' in kwargs, 'The k parameter must be provided'
        assert kwargs['k'] > 0, 'The k parameter must be greater than 0'
        self.k: int = kwargs['k']
        self.name:str = 'CusText'
        self._simWordDict, self._pDict = self._getCustomizedMapping()

    def _getCustomizedMapping(self):
        wordHash:defaultdict = defaultdict(str)
        simWordDict:defaultdict = defaultdict(list)
        pDict:defaultdict = defaultdict(list)
        wordFreq = self.vocab.embeddings.keys()
        for i in range(len(wordFreq)):
            word = wordFreq[i]
            if word not in wordHash:
                indexList = self.euclideanDistance(self.vocab.embeddings[word], self.embMatrix).argsort()[:self.k]
                wordList = [self._index2word[index] for index in indexList]
                embeddingList = np.array([self.vocab.embeddings[w] for w in wordList])
                


    def processText(self,text:List[str])->str:
        '''
        method selfProcessQueryText: this method is used to process the Text and return the obfuscated Text

        : param embs: np.array the embeddings of the words
        : return: str the obfuscated Text
        '''
        ...
        



            

