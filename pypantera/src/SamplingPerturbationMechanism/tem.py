from .AbstractSamplingPerturbationMechanism import AbstractSamplingPerturbationMechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List

class TEM(AbstractSamplingPerturbationMechanism):
    '''
    BibTeX of TEM Mechanism, extends CMP mechanism class of the pypanter package:

    @inbook{doi:10.1137/1.9781611977653.ch99,
        author = {Ricardo Silva Carvalho and Theodore Vasiloudis and Oluwaseyi Feyisetan and Ke Wang},
        title = {TEM: High Utility Metric Differential Privacy on Text},
        booktitle = {Proceedings of the 2023 SIAM International Conference on Data Mining (SDM)},
        chapter = {},
        pages = {883-890},
        doi = {10.1137/1.9781611977653.ch99},
        URL = {https://epubs.siam.org/doi/abs/10.1137/1.9781611977653.ch99},
        eprint = {https://epubs.siam.org/doi/pdf/10.1137/1.9781611977653.ch99},
            abstract = { Abstract Ensuring the privacy of users whose data are used to train Natural Language Processing (NLP) models is necessary to build and maintain customer trust. Differential Privacy (DP) has emerged as the most successful method to protect the privacy of individuals. However, applying DP to the NLP domain comes with unique challenges. The most successful previous methods use a generalization of DP for metric spaces, and apply the privatization by adding noise to inputs in the metric space of word embeddings. However, these methods assume that one specific distance measure is being used, ignore the density of the space around the input, and assume the embeddings used have been trained on public data. In this work we propose Truncated Exponential Mechanism (TEM), a general method that allows the privatization of words using any distance metric, on embeddings that can be trained on sensitive data. Our method makes use of the exponential mechanism to turn the privatization step into a selection problem. This allows the noise applied to be calibrated to the density of the embedding space around the input, and makes domain adaptation possible for the embeddings. In our experiments, we demonstrate that our method outperforms the state-of-the-art in terms of utility for the same level of privacy, while providing more flexibility in the metric selection. }
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
        self.name:str = 'TEM'

        self.candidates: dict = {}
        #self._internalPreprocessing()

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
        return npr.gumbel(gumbel_mean, gumbel_scale, 1)

    def _getLw(self, word:str) -> tuple[List[tuple], List[str]]:
        distance:np.array = self.euclideanDistance(self.vocab.embeddings[word].reshape(1,-1), self.embMatrix)
        #select only distances that are below self.gamma
        indices_Lw:list = np.where(distance[0] <= self.gamma)[0]
        indices_L_hat_w:list = np.where(distance[0] > self.gamma)[0]
        Lw:List[tuple] = [(self._index2word[i], -distance[0][i]) for i in indices_Lw]
        L_hat_w:list = [self.embMatrix[i] for i in indices_L_hat_w]
        if len(L_hat_w) == 0:
            #set value to -inf
            temp_score:float = -np.inf
        else:
            temp_score:float = -self.gamma + 2 *np.log(len(L_hat_w))/self.epsilon
        tempWord:str = 'PLACEHOLDERWORD'
        Lw.append((tempWord, temp_score))
        return Lw, L_hat_w

    def processText(self, 
                 text:List[str]) -> str:
        '''
        processText method: this method is used to process the Text accordingly
        to the definition of the TEM mechanism, see BibTeX ref

        : param embs: np.array the embeddings of the Text
        : return: str the obfuscated Text

        Usage example:
        >>> embPath: str = 'pathToMyEmbeddingsFile.txt'
        >>> eps: float = 0.1
        >>> beta: float = 0.001
        >>> mech1 = TEM({'embPath': embPath, 'epsilon': eps, 'beta': beta})
        >>> words: List[str] = ['what is the capitol of france']
        >>> embs: np.array = mech1.getEmbeddings(words)
        >>> obfuscatedText: str = mech1.processText(embs)
        '''
        #print(f'text: {text}')
        def _processWord(word:str) -> str:
            '''
            _processWord method: this method is used to process the word accordingly to the definition of the TEM mechanism

            : param word: str the word to process
            : return: str the processed obfuscated word
            '''
            if word not in self.vocab.embeddings.keys():
                self.vocab.embeddings[word] = np.zeros(self.embMatrix.shape[1]) + npr.normal(0, 1, self.embMatrix.shape[1]) #handle OoV words
            Lw, L_hat_w = self._getLw(word)
            #add to the scores in Lw the noise
            Lw:List[tuple]= [(w, s + self.pullNoise()) for w, s in Lw]
            #get the word with the highest score
            selectedWord:str = max(Lw, key=lambda x: x[1])[0]
            if selectedWord == 'PLACEHOLDERWORD':
                try:
                    selectedWord:str = np.random.choice(L_hat_w, 1)[0]
                except:
                    selectedWord:str = word
            return selectedWord
        
        finalText:List[str] = list(map(_processWord, text))
        return ' '.join(finalText)
            

        




        




        ...