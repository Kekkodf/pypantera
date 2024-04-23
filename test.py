#from pypantera.src.santext import SanText
#from pypantera.src.wbb import WBB
#from pypantera.src.tem import TEM
import time
import numpy as np
import pandas as pd
import os
from typing import List
import multiprocessing as mp
import argparse
import logging
from joblib import Parallel, delayed

from pypantera.src.mechanism import Mechanism
from pypantera.src.cmp import CMP
from pypantera.src.mahalanobis import Mahalanobis
from pypantera.src.vickrey import VickreyCMP, VickreyMhl
from pypantera.src.wbb import WBB
from pypantera.src.custext import CusText
#from pypantera.src.santext import SanText
#from pypantera.src.tem import TEM

#def test(logger) -> None:
    #t0: time = time.time()
    #read the embeddings file from site/embPath.txt
    #with open('/ssd2/data/defaverifr/DATA/embeddings/glove/glove.6B.300d.txt', 'r') as f:
    #    embPath: str = f.readline().strip()
    #print(f"Embeddings file path: {embPath}")
    

    #initialization of the mechanisms
    #mech1: Mechanism = Mechanism({'embPath': embPath, 'epsilon': 40})
    #important notes: The text should be on lowercase and the words should be separated by a space
    #texts = ['what is the capitol of france', 'who stole the bread', 'how many times does the cat jump on the table']

    #---------------------CMP---------------------
    #mech1: CMP = CMP({'embPath': embPath, 'epsilon':4})
    #obfuscatedText = mech1.obfuscateText(text, 1)
    #print(f"Obfuscated text: {obfuscatedText}")

    #---------------------Mahalanobis---------------------
    #mech1: Mahalanobis = Mahalanobis({'embPath': embPath, 'epsilon': 0.1, 'lambda': 1})
    #obfuscatedText = mech1.obfuscateText(text, 1)
    #print(f"Obfuscated text: {obfuscatedText}")

    #---------------------Vickrey---------------------
    #mech1: VickreyCMP = VickreyCMP({'embPath': embPath, 'epsilon': 0.1, 't': 0.5})
    #obfuscatedText = mech1.obfuscateText(text, 1)
    #print(f"Obfuscated text: {obfuscatedText}")
    #mech1: VickreyMhl = VickreyMhl({'embPath': embPath, 'epsilon': 0.1, 'lambda': 1, 't': 0.5})
    #obfuscatedText = mech1.obfuscateText(text, 1)
    #print(f"Obfuscated text: {obfuscatedText}")
    #---------------------WBB---------------------
    #mech1: CusText = CusText({'embPath': embPath, 'epsilon': 5, 'k': 4}) 
    #for text in texts:
    #    text = text.lower().split()
    #    obfuscatedText = mech1.tokenMappingGeneration(text)
    #    print(f"Obfuscated text: {obfuscatedText}")
    
    

    #df: pd.DataFrame = pd.DataFrame(columns=['text', 'obfuscatedText'])
    #read the queries from the file
    #df['text'] = queries
    #df['text'] = ['what is the capitol of france', 'what is the capitol of germany', 'what is the capitol of italy']
    #df['obfuscatedText'] = df['text'].apply(lambda x: mech1.obfuscateText(x, 1))
    ##in the obfuscated text column we have a list of obfuscated queries, create new lines in the df for each query
    #df = df.explode('obfuscatedText')
    #df['mechanism'] = mech1.__class__.__name__
    ##save the obfuscated text to a csv file
    #if not os.path.exists('test/obfuscated'):
    #    os.makedirs('test/obfuscated')
    #df.to_csv(f'test/obfuscated/obfuscatedTestRun_{mech1.__class__.__name__}_{mech1.epsilon}.csv', index=False)
    #print(f"Time taken: {time.time() - t0}")
    

if __name__ == '__main__':
    #create a logger
    if os.path.exists('./pypantera/logs/logger.log'):
        os.remove('./pypantera/logs/logger.log')
    else:
        os.mkdir('./pypantera/logs')
    
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='./pypantera/logs/logger.log', 
                        level=logging.INFO,
                        format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.info('Starting the obfuscation process...')
    #define the arguments
    embPath: str = '/ssd2/data/defaverifr/DATA/embeddings/glove/glove.6B.300d.txt'
    epsilons: List[float] = [0.01, 0.1, 1, 5, 10, 12.5, 15, 17.5, 20, 50]
    queriesPath: str = '/ssd2/data/defaverifr/DATA/queries/msmarco/trec-dl-19.csv'
    logger.info(f"Embeddings file path: {embPath}")
    logger.info(f"Queries file path: {queriesPath}")
    logger.info(f"Epsilons: {epsilons}")
    mechanisms = [Mahalanobis({'embPath': embPath, 'epsilon': e, 'lambda': 1}) for e in epsilons]
    logger.info(f"Initialized one mechanism of type {mechanisms[0].__class__.__name__} for each epsilon value")
    #define iterables
    data = pd.read_csv(queriesPath, sep = ',')
    #tasks = [(mech.obfuscateText, data, 20) for mech in mechanisms]
    num_cores = mp.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(mech.obfuscateText)(data, 20) for mech in mechanisms)
    #results is a list of pandas dataframe
    logger.info(f"Obfuscated the queries for each mechanism")
    
    for i, df in enumerate(results):
        df = df.explode('obfuscatedText')
        df['mechanism'] = mechanisms[i].__class__.__name__
        df['epsilon'] = mechanisms[i].epsilon
        #save the obfuscated text to a csv file
        if not os.path.exists('./pypantera/results'):
            os.makedirs('./pypantera/results')
        df.to_csv(f'./pypantera/results/obfuscated_{mechanisms[i].__class__.__name__}_{mechanisms[i].epsilon}.csv', index=False)
        logger.info(f"Saved the obfuscated queries to a csv file for mechanism {mechanisms[i].__class__.__name__} with epsilon {mechanisms[i].epsilon}")
        #df = pd.DataFrame(columns=['id', 'text', 'obfuscatedText', 'epsilon', 'mechanism'])
        
    

    ...