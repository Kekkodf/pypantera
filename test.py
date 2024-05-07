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

from pypantera.src.mechanism import Mechanism
from pypantera.src.utils.helper import createLogger, selectMechanism, saveResults
    

if __name__ == '__main__':

    #create a logger
    logger:object = createLogger()

    #define the arguments parser
    parser:object = argparse.ArgumentParser(description='Obfuscate texts using different mechanisms')
    parser.add_argument('--task', '-t', type=str, help='The task to perform', default='retrieval')
    parser.add_argument('--embPath', '-eP', type=str, help='The path to the embeddings file', default='/ssd2/data/defaverifr/DATA/embeddings/glove/glove.6B.300d.txt')
    parser.add_argument('--inputPath', '-i', type=str, help='The path to the input file', default='/ssd2/data/defaverifr/DATA/queries/msmarco/trec-dl-19.csv')
    parser.add_argument('--mechanism', '-m', type=str, help='The mechanism to use', default='Mahalanobis')
    parser.add_argument('--epsilons','-e', type=float, help='The list of epsilon values to use', nargs='+', default=[1, 5, 10, 12.5, 15, 17.5, 20, 50])
    parser.add_argument('--numberOfObfuscations', '-n', type=int, help='The number of obfuscations to perform', default=5)
    args:object = parser.parse_args()

    #log the arguments
    logger.info(f"Task to perform: {args.task}")
    logger.info(f"Embeddings file path: {args.embPath}")
    logger.info(f"Input file path: {args.inputPath}")
    logger.info(f"Mechanism to use: {args.mechanism}")
    logger.info(f"Epsilon values to use: {args.epsilons}")    
        
    #initialize the mechanisms
    mechanisms:List[Mechanism] = selectMechanism(args, logger)
    logger.info('Starting the obfuscation process...')

    #define iterable
    data:pd.DataFrame = pd.read_csv(args.inputPath, sep = ',')

    #obfuscate the queries using multiprocessing
    num_cores:int = mp.cpu_count()
    
    with mp.Pool(num_cores) as pool:
        results:List[pd.DataFrame] = pool.starmap(Mechanism.obfuscateText, [(mech, data, args.numberOfObfuscations) for mech in mechanisms])
    
    logger.info(f"Obfuscation finished! {len(results)} results obtained (one for each epsilon).")

    #results is a list of pandas dataframe, each dataframe contains the obfuscated queries for a specific epsilon value
    logger.info('Saving the obfuscated queries to a csv file...')
    
    #save the results to a csv file
    saveResults(results, mechanisms, args, logger)
    logger.info("Obfuscation process completed successfully!")