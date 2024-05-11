from .AbstractSamplingPerturbationMechanism import AbstractSamplingPerturbationMechanism
import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
import multiprocessing as mp
from typing import List

class TEM(AbstractSamplingPerturbationMechanism):
    '''
    BibTeX of TEM Mechanism, extends CMP mechanism class of the pypanter package:

    @inproceedings{DBLP:conf/sdm/CarvalhoVF023,
        author       = {Ricardo Silva Carvalho and
                         Theodore Vasiloudis and
                         Oluwaseyi Feyisetan and
                         Ke Wang},
         editor       = {Shashi Shekhar and
                         Zhi{-}Hua Zhou and
                         Yao{-}Yi Chiang and
                         Gregor Stiglic},
         title        = {{TEM:} High Utility Metric Differential Privacy on Text},
         booktitle    = {Proceedings of the 2023 {SIAM} International Conference on Data Mining,
                         {SDM} 2023, Minneapolis-St. Paul Twin Cities, MN, USA, April 27-29,
                         2023},
         pages        = {883--890},
         publisher    = {{SIAM}},
         year         = {2023},
         url          = {https://doi.org/10.1137/1.9781611977653.ch99},
         doi          = {10.1137/1.9781611977653.CH99},
         timestamp    = {Tue, 17 Oct 2023 16:40:14 +0200},
         biburl       = {https://dblp.org/rec/conf/sdm/CarvalhoVF023.bib},
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
        finalText:List[str] = []
        for word in text:
            if word not in self.vocab.embeddings.keys():
                self.vocab.embeddings[word] = np.zeros(self.embMatrix.shape[1]) + npr.normal(0, 1, self.embMatrix.shape[1])
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
            finalText.append(selectedWord)
        return ' '.join(finalText)
            

        




        




        ...