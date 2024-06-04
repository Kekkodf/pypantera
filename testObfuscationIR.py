import numpy as np
import pandas as pd
from typing import List
import multiprocessing as mp

from pypantera.src.AbstractTextObfuscationDPMechanism import AbstractTextObfuscationDPMechanism
from pypantera.src.utils.helper import createLogger, createParser, selectMechanism, saveResults
    

if __name__ == '__main__':

    #create a logger
    logger:object = createLogger()

    #define the arguments parser
    parser = createParser()
    args:object = parser.parse_args()
    
    #log the arguments
    logger.info(f"Task to perform: {args.task}")
    logger.info(f"Embeddings file path: {args.embPath}")
    logger.info(f"Input file path: {args.inputPath}")
    logger.info(f"Mechanism to use: {args.mechanism}")
    logger.info(f"Epsilon values to use: {args.epsilons}")    
        
    #initialize the mechanisms
    mechanisms:List[AbstractTextObfuscationDPMechanism] = selectMechanism(args, logger)
    logger.info('Starting the obfuscation process...')
    
    #define iterable
    data:pd.DataFrame = pd.read_csv(args.inputPath, sep = ',')

    #obfuscate the queries using multiprocessing
    num_cores:int = mp.cpu_count()
    
    with mp.Pool(num_cores) as pool:
        results:List[pd.DataFrame] = pool.starmap(AbstractTextObfuscationDPMechanism.obfuscateText, [(mech, data, args.numberOfObfuscations) for mech in mechanisms])
    
    logger.info(f"Obfuscation finished! {len(results)} results obtained (one for each epsilon).")

    #results is a list of pandas dataframe, each dataframe contains the obfuscated queries for a specific epsilon value
    logger.info('Saving the obfuscated queries to a csv file...')
    
    #save the results to a csv file
    saveResults(results, mechanisms, args, logger)
    logger.info("Program terminated successfully!")