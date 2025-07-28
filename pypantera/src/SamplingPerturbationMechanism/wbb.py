import numpy as np
from .AbstractSamplingPerturbationMechanism import AbstractSamplingPerturbationMechanism
import nltk
from nltk import pos_tag

desired_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS']

class WBB(MechaAbstractSamplingPerturbationMechanismnism):

    def __init__(self, kwargs):
        super().__init__()
        self.epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 1.0
        self.k = int(kwargs['k'])
        self.n = int(kwargs['n'])
        _, self.m = self.emb_matrix.shape
        self.desired_pos_tags = desired_pos_tags
        print(f"WBB mechansim initialized with POS tags: {self.desired_pos_tags}")
        self.measure = str(input("Enter the measure for the WBB mechanism (angle, euclidean, product): "))
        print('----------------')

    def euclidean_distance_matrix(self, x, y):
                x_expanded = x[:, np.newaxis, :]  # Shape: [n, 1, d]
                y_expanded = y[np.newaxis, :, :]  # Shape: [1, m, d]
                return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))

    def cosine_distance_matrix(self, x, y):
                x_expanded = x[:, np.newaxis, :]
                y_expanded = y[np.newaxis, :, :]
                return 1 - np.sum(x_expanded * y_expanded, axis=2) / (np.linalg.norm(x_expanded, axis=2) * np.linalg.norm(y_expanded, axis=2))

    def mapping_function(self, emb_matrix):
        if self.measure == 'angle':
            distance = self.cosine_distance_matrix(emb_matrix, self.emb_matrix)
            closest = np.argsort(distance, axis=1)[:, self.k:self.n]
            #closest gives the indices of the words in the vocab that are at position k+1 to n
            candidates = []
            for i in range(len(closest)):
                new_word_candidates = [self.idx2word[word_index] for word_index in closest[i]]
                candidates.append(new_word_candidates)
            #get the distances of the closest words
            distance = np.array([distance[i][closest[i]] for i in range(len(closest))])
            return candidates, distance
            
            
        elif self.measure == 'euclidean':
            distance = self.euclidean_distance_matrix(emb_matrix, self.emb_matrix)
            closest = np.argsort(distance, axis=1)[:, self.k:self.n]
            candidates = []
            for i in range(len(closest)):
                new_word_candidates = [self.idx2word[word_index] for word_index in closest[i]]
                candidates.append(new_word_candidates)
            #get the distances of the closest words
            distance = np.array([distance[i][closest[i]] for i in range(len(closest))])
            return candidates, distance
        
        elif self.measure == 'product':
            def product_metrix(x, y):
                return self.cosine_distance_matrix(x, y) * self.euclidean_distance_matrix(x, y)
            distance = product_metrix(emb_matrix, self.emb_matrix)
            closest = np.argsort(distance, axis=1)[:, self.k:self.n]
            candidates = []
            for i in range(len(closest)):
                new_word_candidates = [self.idx2word[word_index] for word_index in closest[i]]
                candidates.append(new_word_candidates)
            #get the distances of the closest words
            distance = np.array([distance[i][closest[i]] for i in range(len(closest))])
            return candidates, distance
        
        else:
            raise ValueError("Invalid measure for WBB mechanism. Please select a valid measure from the list when constructing mechanism.")
    
    def noisy_probabilities(self, distance):
        mu = np.mean(distance)
        sigma = np.std(distance)
        scores = [1/(1+np.exp(-self.epsilon*(d-mu)/2*sigma)) for d in distance]
        p = scores/np.sum(scores)
        return p
    
    def obfuscate(self, query):
        ##TOKENIZATION & POS tag
        # Tokenize query
        tokens = nltk.word_tokenize(' '.join(query))
        # POS tag tokens
        pos_tokens = pos_tag(tokens)
        final_qry = []
        emb_matrix = []
        for token in pos_tokens:
            if token[1] in self.desired_pos_tags:
                # If the token's POS tag is desired, add its embedding to emb_matrix
                emb_matrix.append(self.vocab[token[0]])
        ## MAPPING FUNCTION
        emb_matrix = np.array(emb_matrix)
        # Find the n closest words to each word in emb_matrix
        candidates, distance = self.mapping_function(emb_matrix)
        ## NOISY SAMPLING
        #compute the probability of sampling
        probabilities = [self.noisy_probabilities(distance[i]) for i in range(len(candidates))]
        #sample the words
        sampled_obfuscated_word = [np.random.choice(candidates[i], p=probabilities[i]) for i in range(len(candidates))]        
        #replace the words in the original query
        j = 0
        for token in pos_tokens:
            if token[1] in self.desired_pos_tags:
                final_qry.append(sampled_obfuscated_word[j])
                j += 1
            else:
                final_qry.append(token[0])
        return ' '.join(final_qry)
        
