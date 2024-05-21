import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer

from pypantera.src.AbstractTextObfuscationDPMechanism import AbstractTextObfuscationDPMechanism
from pypantera.src.utils import metrics

from tqdm import tqdm
tqdm.pandas()    
import os

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    mechanisms:List[str] = ['CMP', 'Mahalanobis', 'VickreyCMP', 'VickreyMhl', 'SanText', 'CusText', 'TEM']
    epsilons:List[float] = [1, 5, 10, 12.5, 15, 17.5, 20, 50]

    model:SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    #read the datasets in results/retrieval folder
    df_sim = pd.DataFrame(columns=['id','Semantic Similarity', 'Lexical Similarity', 'mechanism', 'epsilon'])
    results = []
    for mech in mechanisms:
        print(f'Processing mechanism: {mech}')
        for e in epsilons:
            print(f'Processing epsilon: {e}')
            df = pd.read_csv(f'./results/retrieval/{mech}/obfuscatedText_{mech}_{e}.csv', sep=',', header=0)
            df_sim['Semantic Similarity'] = df.progress_apply(lambda x: metrics.sentenceSimilarity(model, x['text'], x['obfuscatedText']), axis=1)
            df_sim['Lexical Similarity'] = df.progress_apply(lambda x: metrics.lexicalSimilarity(x['text'], x['obfuscatedText']), axis=1)
            df_sim['id'] = df['id']
            df_sim['mechanism'] = mech
            df_sim['epsilon'] = e
            meanSemSim = df_sim['Semantic Similarity'].mean()
            meanLexSim = df_sim['Lexical Similarity'].mean()
            tuple_mech = (mech, e, meanSemSim, meanLexSim)
            results.append(tuple_mech)
    list_of_results = [{'mechanism': mech, 'epsilon': e, 'Semantic Similarity': meanSemSim, 'Lexical Similarity': meanLexSim} for mech, e, meanSemSim, meanLexSim in results]
    results_df = pd.DataFrame(list_of_results, columns=['mechanism', 'epsilon', 'Semantic Similarity', 'Lexical Similarity'])
    #create a pivot table with the results
    pivotSemSim = results_df.pivot(index='mechanism', columns='epsilon', values='Semantic Similarity')
    pivotLexSim = results_df.pivot(index='mechanism', columns='epsilon', values='Lexical Similarity')
    #save the pivot table to a csv file
    if not os.path.exists('./results/privacy'):
        os.mkdir('./results/privacy')
    pivotSemSim.to_csv('./results/privacy/pivotSemSim.csv')
    pivotLexSim.to_csv('./results/privacy/pivotLexSim.csv')


