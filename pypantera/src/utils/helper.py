import os
import logging
import pandas as pd
from typing import List

from ..AbstractTextObfuscationDPMechanism import AbstractTextObfuscationDPMechanism
#from ..EmbeddingPerturbationMechanism import AbstractEmbeddingPerturbationMechanism
#from ..EmbeddingPerturbationMechanism import AbstractSamplingPerturbationMechanism

from ..EmbeddingPerturbationMechanism.cmp import CMP
from ..EmbeddingPerturbationMechanism.mahalanobis import Mahalanobis
from ..EmbeddingPerturbationMechanism.vickrey import VickreyCMP, VickreyMhl

from ..SamplingPerturbationMechanism.santext import SanText
from ..SamplingPerturbationMechanism.custext import CusText
from ..SamplingPerturbationMechanism.tem import TEM


def createLogger() -> object:
    '''
    creates the logger object and returns it
    Runtime logs can be found in the pypantera/logs/logger.log file
    '''
    if os.path.exists('./pypantera/logs/logger.log'):
        os.remove('./pypantera/logs/logger.log')
    else:
        os.mkdir('./pypantera/logs')
    
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='./pypantera/logs/logger.log', 
                        level=logging.INFO,
                        format=FORMAT)
    logger = logging.getLogger(__name__)
    return logger

def selectMechanism(args:object, logger:object) -> list:
    '''
    select the mechanism to use based on the argument.mechanism provided, initialize a list of mechanisms for different parameters
    '''
    if args.mechanism in ['CMP', 'Mahalanobis', 'VickreyCMP', 'VickreyMhl']:
        if args.mechanism == 'CMP':
            mechanisms = [CMP({'embPath': args.embPath, 'epsilon': e}) for e in args.epsilons]
        elif args.mechanism == 'Mahalanobis':
            mechanisms = [Mahalanobis({'embPath': args.embPath, 'epsilon': e, 'lam': args.lam}) for e in args.epsilons]
            logger.info(f"Lambda: {mechanisms[0].lam}")
        elif args.mechanism == 'VickreyCMP':
            mechanisms = [VickreyCMP({'embPath': args.embPath, 'epsilon': e, 't':args.t}) for e in args.epsilons]
            logger.info(f"Treshold: {mechanisms[0].t}")
        elif args.mechanism == 'VickreyMhl':
            mechanisms = [VickreyMhl({'embPath': args.embPath, 'epsilon': e, 'lam': args.lam, 't': args.t}) for e in args.epsilons]
            logger.info(f"Lambda: {mechanisms[0].lam}")
            logger.info(f"Treshold: {mechanisms[0].t}")
    elif args.mechanism in ['SanText', 'CusText', 'TEM']:
        if args.mechanism == 'SanText':
            mechanisms = [SanText({'embPath': args.embPath, 'epsilon': e}) for e in args.epsilons]
        elif args.mechanism == 'CusText':
            mechanisms = [CusText({'embPath': args.embPath, 'epsilon': e, 'k':args.k}) for e in args.epsilons]
            logger.info(f"K: {mechanisms[0].k}")
        elif args.mechanism == 'TEM':
            mechanisms = [TEM({'embPath': args.embPath, 'epsilon': e, 'beta': args.beta}) for e in args.epsilons]
            logger.info(f"Beta: {mechanisms[0].beta} --> Gamma: {mechanisms[0].gamma}")
    else:
        logger.error('The mechanism provided is not valid. Please choose one of the following mechanisms: CMP, Mahalanobis, VickreyCMP, VickreyMhl, CusText')
        exit(1)
    logger.info(f"Initialized one mechanism of type {mechanisms[0].__class__.__name__} for each epsilon value")
    return mechanisms

def saveResults(results:List[pd.DataFrame], mechanisms:List[AbstractTextObfuscationDPMechanism], args:object, logger:object) -> None:
    '''
    Save results to a csv file in the results/args.task folder
    '''
    for i, df in enumerate(results):
        df:pd.DataFrame = df.explode('obfuscatedText')
        df['mechanism'] = mechanisms[i].__class__.__name__
        df['epsilon'] = mechanisms[i].epsilon
        #save the obfuscated text to a csv file
        if not os.path.exists('./results'):
            os.makedirs('./results')
        if not os.path.exists(f'./results/{args.task}'):
            os.makedirs(f'./results/{args.task}')
        df.to_csv(f'./results/{args.task}/obfuscated_{mechanisms[i].__class__.__name__}_{mechanisms[i].epsilon}.csv', index=False)
        logger.info(f"Saved the obfuscated queries to a csv file for mechanism {mechanisms[i].__class__.__name__} with epsilon {mechanisms[i].epsilon}")
    