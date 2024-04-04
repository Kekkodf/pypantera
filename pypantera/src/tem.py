from .mechanism import Mechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List

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
    def __init__(self, kwargs: dict) -> None:
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
        >>> lam: float = 0.1 #anyvalue of lambda must be between 0 and 1
        >>> mech1 = TEM({'embPath': embPath, 'epsilon': eps, })
        '''
        super().__init__(kwargs)