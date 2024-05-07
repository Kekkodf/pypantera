import os
import logging
import pandas as pd
from typing import List

from pypantera.src.mechanism import Mechanism
from pypantera.src.cmp import CMP
from pypantera.src.mahalanobis import Mahalanobis
from pypantera.src.vickrey import VickreyCMP, VickreyMhl
from pypantera.src.custext import CusText


def createLogger() -> object:
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
    if args.mechanism == 'CMP':
        mechanisms = [CMP({'embPath': args.embPath, 'epsilon': e}) for e in args.epsilons]
    elif args.mechanism == 'Mahalanobis':
        mechanisms = [Mahalanobis({'embPath': args.embPath, 'epsilon': e, 'lambda': 1}) for e in args.epsilons]
    elif args.mechanism == 'VickreyCMP':
        mechanisms = [VickreyCMP({'embPath': args.embPath, 'epsilon': e, 't':0.75}) for e in args.epsilons]
    elif args.mechanism == 'VickreyMhl':
        mechanisms = [VickreyMhl({'embPath': args.embPath, 'epsilon': e, 'lambda': 1, 't': 0.75}) for e in args.epsilons]
    elif args.mechanism == 'CusText':
        mechanisms = [CusText({'embPath': args.embPath, 'epsilon': e}) for e in args.epsilons]
    else:
        logger.error('The mechanism provided is not valid. Please choose one of the following mechanisms: CMP, Mahalanobis, VickreyCMP, VickreyMhl, CusText')
        exit(1)
    logger.info(f"Initialized one mechanism of type {mechanisms[0].__class__.__name__} for each epsilon value")
    return mechanisms

def saveResults(results:List[pd.DataFrame], mechanisms:List[Mechanism], args:object, logger:object) -> None:
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
    