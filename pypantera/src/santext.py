from .mechanism import Mechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List

class SanText(Mechanism):
    '''
    BibTeX of SanText Mechanism, extends CMP mechanism class of the pypanter package:

    @inproceedings{ACL21/YueDu21,
    author    = {Xiang Yue and Minxin Du and Tianhao Wang and Yaliang Li and Huan Sun and Sherman S. M. Chow},
    title     = {Differential Privacy for Text Analytics via Natural Text Sanitization},
    booktitle = {Findings, {ACL-IJCNLP} 2021},
    year      = {2021},
    }
    '''
    def __init__(self, kwargs: dict[str:object]) -> None:
        '''
        Initialization of the SanText Object

        : param kwargs: dict the dictionary containing the parameters of the Mechanism Object 
                        + the specific parameters of the SanText Mechanism ()

        Once the SanText Object is created, the user can use the obfuscateText method to obfuscate 
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
        >>> mech1 = SanText({'embPath': embPath, 'epsilon': eps, })
        '''
        super().__init__(kwargs)