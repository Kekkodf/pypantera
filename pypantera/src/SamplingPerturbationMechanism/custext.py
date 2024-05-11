from .AbstractSamplingPerturbationMechanism import AbstractSamplingPerturbationMechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List
import time
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

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
        assert 'distance' in kwargs, 'The distance parameter must be provided'
        assert kwargs['distance'] in ['euclidean', 'cosine'], 'The distance parameter must be either euclidean or cosine'
        self.distance:str = kwargs['distance']
        self.name:str = 'CusText'
        self._simWordDict = {}
        self._pDict = {}

    def _getCustomizedMapping(self, word) -> tuple[defaultdict, defaultdict]:
        '''
        precompute the mapping of the words and their respective probabilities
        '''
        simWordDict:defaultdict = defaultdict(list)
        pDict:defaultdict = defaultdict(list)
        wordFreq = list(self._simWordDict.keys())
        if self.distance == 'euclidean':
            if word not in wordFreq:
                try:
                    wordEmb = self.vocab.embeddings[word].reshape(1,-1)
                except:
                    wordEmb = np.zeros(self.embMatrix.shape[1]) + npr.normal(0, 1, self.embMatrix.shape[1]).reshape(1,-1)
                    
                indexList = self.euclideanDistance(wordEmb, self.embMatrix).argsort()[:self.k][0]
                wordList = [self._index2word[index] for index in indexList[:self.k]]
                embeddingList = np.array([self.vocab.embeddings[w] for w in wordList])

                simDistList = self.euclideanDistance(wordEmb, embeddingList)[0]
                min_max_dist = max(simDistList) - min(simDistList)
                min_dist = min(simDistList)
                newSimDistList = [-(x-min_dist)/min_max_dist for x in simDistList]
                tmp = [np.exp(self.epsilon*x/2) for x in newSimDistList]
                norm = sum(tmp)
                p = [x/norm for x in tmp]
                pDict[word] = p
                simWordDict[word] = wordList
                self._pDict[word] = p
                self._simWordDict[word] = wordList 
            else:
                simWordDict[word] = self._simWordDict[word]
                pDict[word] = self._pDict[word]

        elif self.distance == 'cosine':
            if word not in wordFreq:
                indexList = self.cosineSimilarity(self.vocab.embeddings[word].reshape(1,-1), self.embMatrix).argsort()[:self.k][0]
                wordList = [self._index2word[index] for index in indexList[:self.k]]
                embeddingList = np.array([self.vocab.embeddings[w] for w in wordList])

                simDistList = self.cosineSimilarity(self.vocab.embeddings[word].reshape(1,-1), embeddingList)[0]
                min_max_dist = max(simDistList) - min(simDistList)
                min_dist = min(simDistList)
                newSimDistList = [(x-min_dist)/min_max_dist for x in simDistList]
                tmp = [np.exp(self.epsilon*x/2) for x in newSimDistList]
                norm = sum(tmp)
                p = [x/norm for x in tmp]
                pDict[word] = p
                simWordDict[word] = wordList
                self._pDict[word] = p
                self._simWordDict[word] = wordList 
            else:
                simWordDict[word] = self._simWordDict[word]
                pDict[word] = self._pDict[word]
        return simWordDict, pDict
        ...    

    #def _generateText(self, text:str) -> str:
    #    ...

    def processText(self, text:List[str])->str:
        ''' 
        method selfProcessQueryText: this method is used to process the Text and return the obfuscated Text

        : param embs: np.array the embeddings of the words
        : return: str the obfuscated Text
        '''
        finalText:list = []
        for word in text:
            simWordDict, pDict = self._getCustomizedMapping(word)
            newWord:str = np.random.choice(simWordDict[word], 1, p=pDict[word])[0]
            finalText.append(newWord)
        return ' '.join(finalText)
            
        ...
        



            

