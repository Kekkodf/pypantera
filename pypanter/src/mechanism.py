import numpy as np
import numpy.random as npr
import multiprocessing as mp
import utils
"""
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
    """

class mechanism():
    
    def __init__(self, kwargs):
        self.info = "This is the base mechanism class. The base mechanism is the implemantation of the CMP mechansim (Feyisetan et al., 2020)."
        self.vocab, self.embMatrix, self.index2word, self.word2index = utils.vocab(kwargs['embPath'])
        self.epsilon = kwargs['epsilon']

    #get method
    def getAttributes(self):
        return self.info, self.vocab, self.embMatrix, self.index2word, self.word2index, self.epsilon

    #multiCoreRunner method: this method is used to run the obfuscation process in parallel
    def multiCoreRunner(self):
        return
    
    #runner method: this method is used to run the obfuscation process in a single core
    def runner(self):
        return
    
    #obfuscateText method: this method is used to obfuscate the text
    def obfuscateText(self):
        return
    
    #pullNoise method: this method is used to pull noise accordingly to the definition of the CMP mechanism
    def pullNoise(self):
        N = self.epsilon * npr.multivariate_normal(np.zeros(self.embMatrix.shape[1]), np.eye(self.embMatrix.shape[1]))
        X = N / np.sqrt(np.sum(N ** 2))
        Y = npr.gamma(self.embMatrix.shape[1], 1 / self.epsilon)
        Z = Y * X
        return Z
    
    def obfuscate_text(self, text, e):
        #text is a list of words
        embs = []
        for word in text:
            if word not in self.vocab.keys():
                embs.append(np.zeros(self.emb_matrix.shape[1]) + npr.normal(0, 1, self.emb_matrix.shape[1]))
            else:
                embs.append(self.vocab[word])
        embs = np.array(embs)
        #add noise to emeddings to each row
        noise = np.array([self.noise_sampling(e) for i in range(len(embs))])
        #logger.info(f"Adding noise to embeddings")
        noisy_emb = embs + noise
        #compute the distance between the noisy embeddings and the matrix of embeddings
        def euclidean_distance_matrix(x, y):
            x_expanded = x[:, np.newaxis, :]
            y_expanded = y[np.newaxis, :, :]
            return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))
        distance = euclidean_distance_matrix(noisy_emb, self.emb_matrix)
        #use argapartition to find the first closest words
        closest = np.argpartition(distance, 1, axis=1)[:, :1]
        final_qry = []
        try:
            for i in range(len(text)):
                final_qry.append(list(self.vocab.keys())[closest[i][0]])
        except:
            final_qry.append(text)
        return final_qry
    
    def worker(self, query, e, number_of_desired_queries):
        query['obfuscated text'] = query['text'].apply(lambda x: x.split())
        #multiply the number of queries
        query = query.loc[query.index.repeat(number_of_desired_queries)].reset_index(drop=True)
        query['obfuscated text'] = query['obfuscated text'].apply(lambda x: self.obfuscate_text(x, e))
        query['obfuscated text'] = query['obfuscated text'].apply(lambda x: " ".join(x))
        #add a column with the value of epsilon
        query['epsilon'] = e
        return query
    
    def obfuscate(self, query, logger, dict_params):
        logger.info(f"Starting obfuscation process")
        with mp.Pool(30) as p:
            tasks = [(query, e, dict_params['number_of_desired_queries']) for e in self.epsilon]
            results = p.starmap(self.worker, tasks)
        logger.info(f"Obfuscation process finished")
        return results
        