import numpy as np
import numpy.random as npr
from .AbstractEmbeddingPerturbationMechanism import AbstractEmbeddingPerturbationMechanism
from typing import List

'''
# CMP Mechanism
'''

class CMP(AbstractEmbeddingPerturbationMechanism):
    '''
    # CMP
    
    BibTeX of CMP Mechanism, extends Mechanism mechanism class of the pypantera package:

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
    
    def __init__(self, kwargs: dict[str:object]) -> None:

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
        super().__init__(kwargs)
        self.name: str = 'CMP'
        

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

        N: np.array = self.epsilon * npr.multivariate_normal(
            np.zeros(self.embMatrix.shape[1]),
            np.eye(self.embMatrix.shape[1])) #pull noise from a multivariate normal distribution
        X: np.array = N / np.sqrt(np.sum(N ** 2)) #normalize the noise
        Y: np.array = npr.gamma(
            self.embMatrix.shape[1],
            1 / self.epsilon) #pull gamma noise
        Z: np.array = Y * X #compute the final noise
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