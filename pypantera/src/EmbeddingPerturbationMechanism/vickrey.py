from .AbstractEmbeddingPerturbationMechanism import AbstractEmbeddingPerturbationMechanism
from .cmp import CMP
from .mahalanobis import Mahalanobis
import numpy as np
import numpy.random as npr
from typing import List

class AbstractVickrey(AbstractEmbeddingPerturbationMechanism):
    '''
    Vickrey Mechanism
    '''
    def __init__(self, kwargs: dict[str:object]) -> None:
        '''
        # Vickrey
        BibTeX of Vickrey Mechanism, extends Mechanism class of the pypanter package:

        @inproceedings{xu-etal-2021-utilitarian,
            title = "On a Utilitarian Approach to Privacy Preserving Text Generation",
            author = "Xu, Zekun  and
              Aggarwal, Abhinav  and
              Feyisetan, Oluwaseyi  and
              Teissier, Nathanael",
            editor = "Feyisetan, Oluwaseyi  and
              Ghanavati, Sepideh  and
              Malmasi, Shervin  and
              Thaine, Patricia",
            booktitle = "Proceedings of the Third Workshop on Privacy in Natural Language Processing",
            month = jun,
            year = "2021",
            address = "Online",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2021.privatenlp-1.2",
            doi = "10.18653/v1/2021.privatenlp-1.2",
            pages = "11--20",
            abstract = "Differentially-private mechanisms for text generation typically add carefully calibrated noise to input words and use the nearest neighbor to the noised input as the output word. When the noise is small in magnitude, these mechanisms are susceptible to reconstruction of the original sensitive text. This is because the nearest neighbor to the noised input is likely to be the original input. To mitigate this empirical privacy risk, we propose a novel class of differentially private mechanisms that parameterizes the nearest neighbor selection criterion in traditional mechanisms. Motivated by Vickrey auction, where only the second highest price is revealed and the highest price is kept private, we balance the choice between the first and the second nearest neighbors in the proposed class of mechanisms using a tuning parameter. This parameter is selected by empirically solving a constrained optimization problem for maximizing utility, while maintaining the desired privacy guarantees. We argue that this empirical measurement framework can be used to align different mechanisms along a common benchmark for their privacy-utility tradeoff, particularly when different distance metrics are used to calibrate the amount of noise added. Our experiments on real text classification datasets show up to 50{\%} improvement in utility compared to the existing state-of-the-art with the same empirical privacy guarantee.",
        }
        '''
        super().__init__(kwargs)
        assert 't' in kwargs, 'The t parameter must be provided and between 0 and 1'
        assert kwargs['t'] >= 0 and kwargs['t'] <= 1, 'The t parameter must be between 0 and 1'
        self.t: float = kwargs['t']
        self.name: str = 'Vickrey'

class VickreyCMP(CMP, AbstractVickrey):
    def __init__(self, kwargs) -> None:
           
            '''
            Initialization of the VickreyCMP Object
            : param kwargs: dict the dictionary containing the parameters of the Mechanism Object 
                            + the specific parameters of the VickreyCMP Mechanism (lambda)
            Once the VickreyCMP Object is created, the user can use the obfuscateText method to obfuscate 
            the text of the provided text.
            The attributes of the Mechanism Object are:
            - vocab: the vocabulary object containing the embeddings
            - embMatrix: the matrix containing the embeddings
            - index2word: the dictionary containing the index to word mapping
            - word2index: the dictionary containing the word to index mapping
            - epsilon: the epsilon parameter of the mechanism
            - t: the t parameter of the Vickrey mechanism
            Usage example:
            >>> embPath: str = 'pathToMyEmbeddingsFile.txt'
            >>> eps: float = 0.1 #anyvalue of epsilon must be greater than 0
            >>> t: float = 0.5 #anyvalue of t must be between 0 and 1
            >>> mech1 = Vickrey.Mhl({'embPath': embPath, 'epsilon': eps, 't': t})
            '''
            super().__init__(kwargs)
            self.name:str = 'VickreyCMP'

    def processText(self, 
                 embs: np.array) -> str:
        '''
        method processText: this method is used to process the Text accordingly
        to the definition of the Vickrey mechanism, see BibTeX ref

        : param embs: np.array the embeddings of the Text
        : return: str the obfuscated Text

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)
        >>> embPath: str = 'pathToMyEmbeddingsFile.txt'
        >>> eps: float = 0.1
        >>> lam: float = 0.1
        >>> t: float = 0.5

        >>> mech1 = Vickrey.Mhl({'embPath': embPath, 'epsilon': eps, 'lambda': lam, 't': t})
        >>> Text: str = 'what is the capitol of france'
        >>> embs: np.array = mech1.getEmbeddings(Text)
        >>> obfuscatedText: str = mech1.processText(embs)
        '''

        length: int = len(embs) #compute number of words

        distance: np.array = self.euclideanDistance(embs, self.embMatrix) #compute distances between words and embeddings in the vocabulary
        closest: np.array = np.argpartition(distance, 2, axis=1)[:, :2] #find the two closest embeddings for each word
        
        distToClosest: np.array = distance[np.tile(np.arange(length).reshape(-1,1),2), closest] #compute the distances to the two closest embeddings
        p = ((1- self.t) * distToClosest[:,1]) / (self.t * distToClosest[:,0] + (1 - self.t) * distToClosest[:, 1]) #compute the probabilities of choosing the second closest embedding
        vickreyChoice: np.array = np.array([npr.choice(2, p=[p[w], 1-p[w]]) for w in range(length)]) #choose the closest embedding according to the probabilities
        noisyEmbeddings: np.array = self.embMatrix[closest[np.arange(length), vickreyChoice]] #get the noisy embeddings

        distance = self.euclideanDistance(noisyEmbeddings, self.embMatrix)
        found = np.argpartition(distance, 1, axis=1)[:, :1]
        
        finalText = self.indexes2words(found)       
        return ' '.join(finalText)
        
    
class VickreyMhl(Mahalanobis, AbstractVickrey):
    def __init__(self, kwargs) -> None:
        '''
        Initialization of the VickreyMahalanobis Object 
        : param kwargs: dict the dictionary containing the parameters of the Mechanism Object 
                        + the specific parameters of the VickreyMahalanobis Mechanism (lambda)  

        Once the VickreyMahalanobis Object is created, the user can use the obfuscateText method to obfuscate 
        the text of the provided text.  

        The attributes of the Mechanism Object are:
        - vocab: the vocabulary object containing the embeddings
        - embMatrix: the matrix containing the embeddings
        - index2word: the dictionary containing the index to word mapping
        - word2index: the dictionary containing the word to index mapping
        - epsilon: the epsilon parameter of the mechanism
        - lam: the lambda parameter of the Mahalanobis mechanism
        - sigma_loc: parameter used for pulling noise in obfuscation
        - t: the t parameter of the Vickrey mechanism   

        Usage example:
        >>> embPath: str = 'pathToMyEmbeddingsFile.txt'
        >>> eps: float = 0.1 #anyvalue of epsilon must be greater than 0
        >>> lam: float = 0.1 #anyvalue of lambda must be between 0 and 1
        >>> t: float = 0.5 #anyvalue of t must be between 0 and 1
        >>> mech1 = Vickrey.Mhl({'embPath': embPath, 'epsilon': eps, 'lambda': lam, 't': t})
        '''
        super().__init__(kwargs)
        self.name:str = 'VickreyMhl'

    def processText(self, 
                 embs: np.array) -> str:
        '''
        method processText: this method is used to process the Text accordingly
        to the definition of the Vickrey mechanism, see BibTeX ref

        : param embs: np.array the embeddings of the Text
        : return: str the obfuscated Text

        Usage example:
        (Considering that the Mechanism Object mech1 has been created
        as in the example of the __init__ method)
        >>> embPath: str = 'pathToMyEmbeddingsFile.txt'
        >>> eps: float = 0.1
        >>> lam: float = 0.1
        >>> t: float = 0.5

        >>> mech1 = Vickrey.Mhl({'embPath': embPath, 'epsilon': eps, 'lambda': lam, 't': t})
        >>> text: str = 'what is the capitol of france'
        >>> embs: np.array = mech1.getEmbeddings(text)
        >>> obfuscatedText: str = mech1.processText(embs)
        '''

        length: int = len(embs) #compute number of words

        distance: np.array = self.euclideanDistance(embs, self.embMatrix) #compute distances between words and embeddings in the vocabulary
        closest: np.array = np.argpartition(distance, 2, axis=1)[:, :2] #find the two closest embeddings for each word
        
        distToClosest: np.array = distance[np.tile(np.arange(length).reshape(-1,1),2), closest] #compute the distances to the two closest embeddings
        p = ((1- self.t) * distToClosest[:,1]) / (self.t * distToClosest[:,0] + (1 - self.t) * distToClosest[:, 1]) #compute the probabilities of choosing the second closest embedding
        vickreyChoice: np.array = np.array([npr.choice(2, p=[p[w], 1-p[w]]) for w in range(length)]) #choose the closest embedding according to the probabilities
        noisyEmbeddings: np.array = self.embMatrix[closest[np.arange(length), vickreyChoice]] #get the noisy embeddings
        
        distance = self.euclideanDistance(noisyEmbeddings, self.embMatrix)
        found = np.argpartition(distance, 1, axis=1)[:, :1]
        
        finalText = self.indexes2words(found)
       
        return ' '.join(finalText)
        