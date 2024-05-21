import os
import logging
import pandas as pd
import argparse
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

'''
helper module: Provides helper functions to create the parser, the logger and to select the mechanism to use


'''

def createParser() -> object:
    '''
    # createParser()
    creates the parser object and returns it
    The arguments should be passed by command line otherwise the program will crush if no embPath and inputPath are provided

    # Required Params:
    - embPath: the path to the embeddings file
    - inputPath: the path to the input file (.csv file)

    # Default Params :
    - mechanism: the mechanism to use (default: CMP)
    - task: the task to compleate
    - t: the treshold value to use Vickrey mechanisms (default: 0.75)
    - beta: the beta value to use for the TEM mechanism (default: 0.001)
    - lam: the lambda value to use for the Mahalanobis and Vickrey mechanisms (default: 1)
    - k: the number of words to sample for the CusText mechanism (default: 5)
    - epsilons: the list of epsilon values to use (default: [1, 5, 10, 12.5, 15, 17.5, 20, 50])
    - numberOfObfuscations: the number of obfuscations to perform (default: 1)
    '''
    parser:object = argparse.ArgumentParser(description='Obfuscate texts using different mechanisms')

    #mechanism params
    parser.add_argument('--task', '-tk', type=str, help='The task to perform', default='sentimentAnalysis')
    parser.add_argument('--embPath', '-eP', type=str, help='The path to the embeddings file', default='../DATA/embeddings/glove/glove.6B.300d.txt')
    parser.add_argument('--inputPath', '-i', type=str, help='The path to the input file', default='../DATA/sentimentAnalysis/twitter/twitter_parsed.csv')
    parser.add_argument('--outputPath', '-o', type=str, help='The path to the output file')
    parser.add_argument('--mechanism', '-m', type=str, help='The mechanism to use', default='CMP', choices=['CMP', 'Mahalanobis', 'VickreyCMP', 'VickreyMhl', 'CusText', 'SanText', 'TEM'])
    parser.add_argument('--t', '-t', type=float, help='The treshold value to use Vickrey mechanisms', default=0.75)
    parser.add_argument('--beta', '-beta', type=float, help='The beta value to use for the TEM mechanism', default=0.001)
    parser.add_argument('--lam', '-lam', type=float, help='The lambda value to use for the Mahalanobis and Vickrey mechanisms', default=1)
    parser.add_argument('--k', '-k', type=int, help='The number of words to sample for the CusText mechanism', default=10)
    parser.add_argument('--distance', '-d', type=str, help='The distance metric to use for the CusText mechanism', default='euclidean')
    parser.add_argument('--epsilons','-e', type=float, help='The list of epsilon values to use', nargs='+', default=[1, 5, 10, 12.5, 15, 17.5, 20, 50])
    parser.add_argument('--numberOfObfuscations', '-n', type=int, help='The number of obfuscations to perform', default=1)
    return parser

def createLogger() -> object:
    '''
    # createLogger()
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
            mechanisms = [CusText({'embPath': args.embPath, 'epsilon': e, 'k':args.k, 'distance':args.distance}) for e in args.epsilons]
            logger.info(f"K: {mechanisms[0].k}")
            logger.info(f"Distance: {mechanisms[0].distance}")
        elif args.mechanism == 'TEM':
            mechanisms = [TEM({'embPath': args.embPath, 'epsilon': e, 'beta': args.beta}) for e in args.epsilons]
            logger.info(f"Beta: {mechanisms[0].beta}")
            for i in range(len(mechanisms)):
                logger.info(f"For mechansim with epsilon = {mechanisms[i].epsilon}, Gamma computed: {mechanisms[i].gamma}")
    else:
        logger.error('The mechanism provided is not valid. Please choose one of the following mechanisms: CMP, Mahalanobis, VickreyCMP, VickreyMhl, CusText')
        exit(1)
    logger.info(f"Initialized one mechanism of type {mechanisms[0].__class__.__name__} for each epsilon value")
    return mechanisms

def saveResults(results:List[pd.DataFrame], mechanisms:List[AbstractTextObfuscationDPMechanism], args:object, logger:object, sentiment, path:str = None) -> None:
    '''
    Save results to a csv file in the results/args.task folder
    '''
    for i, df in enumerate(results):
        df:pd.DataFrame = df.explode('obfuscatedText')
        df['mechanism'] = mechanisms[i].__class__.__name__
        df['epsilon'] = mechanisms[i].epsilon
        df['sentiment'] = sentiment
        #save the obfuscated text to a csv file
        output_path = args.outputPath
        if path == None:
            if not os.path.exists('./results'):
                os.makedirs('./results')
            if not os.path.exists(f'./results/{args.task}'):
                os.makedirs(f'./results/{args.task}')
            if not os.path.exists(f'./results/{args.task}/{mechanisms[i].__class__.__name__}'):
                os.makedirs(f'./results/{args.task}/{mechanisms[i].__class__.__name__}')
            df.to_csv(f'./results/{args.task}/{mechanisms[i].__class__.__name__}/obfuscatedText_{mechanisms[i].__class__.__name__}_{mechanisms[i].epsilon}.csv', index=False)
            logger.info(f"Saved the obfuscated queries to a csv file for mechanism {mechanisms[i].__class__.__name__} with epsilon {mechanisms[i].epsilon}")
        else:
            df.to_csv(path, index=False)
            logger.info(f"Saved the obfuscated queries to a csv file for mechanism {mechanisms[i].__class__.__name__} with epsilon {mechanisms[i].epsilon}")
    logger.info('Obfuscation process completed successfully!')
    os.system(f'cp ./pypantera/logs/logger.log ./results/{args.task}/{mechanisms[0].__class__.__name__}/logger_{mechanisms[0].__class__.__name__}.log')
        