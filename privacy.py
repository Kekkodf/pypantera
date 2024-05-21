import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer

from pypantera.src.utils import metrics

from tqdm import tqdm
tqdm.pandas()    
import os

'''
Script to evaluate the privacy of the obfuscated text using the SentenceTransformer model

The script reads the obfuscated text from the results/retrieval folder and 
computes the semantic and lexical similarity between the original text and the obfuscated text.

The script uses the SentenceTransformer model to compute the sentence similarity and the lexical 
similarity adopting the functions implemented in pyPANTERA.

At the end the results are stored in a pivot table and saved in the results/privacy folder.
'''

if __name__ == '__main__':

    #define the lists of epsilons and mechanism that you want to evaluate
    mechanisms:List[str] = ['CMP', 'Mahalanobis', 'VickreyCMP', 'VickreyMhl', 'SanText', 'CusText', 'TEM']
    epsilons:List[float] = [1, 5, 10, 12.5, 15, 17.5, 20, 50]

    #initialize the SentenceTransformer model
    model:SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')

    #read the datasets in results/retrieval folder
    df_sim:pd.DataFrame = pd.DataFrame(columns=['id','Semantic Similarity', 'Lexical Similarity', 'mechanism', 'epsilon'])
    results:list = []
    #iterated over all the mechanisms and epsilons
    for mech in mechanisms:
        print(f'Processing mechanism: {mech}')
        for e in epsilons:
            print(f'Processing epsilon: {e}')
            #read data
            df:pd.DataFrame = pd.read_csv(f'./results/retrieval/{mech}/obfuscatedText_{mech}_{e}.csv', sep=',', header=0)
            #compute the similarity between the original text and the obfuscated text
            df_sim['Semantic Similarity'] = df.progress_apply(lambda x: metrics.sentenceSimilarity(model, x['text'], x['obfuscatedText']), axis=1)
            df_sim['Lexical Similarity'] = df.progress_apply(lambda x: metrics.lexicalSimilarity(x['text'], x['obfuscatedText']), axis=1)
            #finish the construction of the dataframe (if you want to save the results or see what you're doing)
            df_sim['id'] = df['id']
            df_sim['mechanism'] = mech
            df_sim['epsilon'] = e
            #format the results with 3 decimal places
            meanSemSim:float = round(df_sim['Semantic Similarity'].mean(), 3)
            meanLexSim:float = round(df_sim['Lexical Similarity'].mean(), 3)
            #save the results computed
            tuple_mech:tuple = (mech, e, meanSemSim, meanLexSim)
            results.append(tuple_mech)
    #prepare data for the pivot tables
    list_of_results:List[dict] = [{'mechanism': mech, 'epsilon': e, 'Semantic Similarity': meanSemSim, 'Lexical Similarity': meanLexSim} for mech, e, meanSemSim, meanLexSim in results]
    results_df:pd.DataFrame = pd.DataFrame(list_of_results, columns=['mechanism', 'epsilon', 'Semantic Similarity', 'Lexical Similarity'])
    #create a pivot table with the results
    pivotSemSim:pd.DataFrame = results_df.pivot(index='mechanism', columns='epsilon', values='Semantic Similarity')
    pivotLexSim:pd.DataFrame = results_df.pivot(index='mechanism', columns='epsilon', values='Lexical Similarity')
    #save the pivot table to a csv file
    if not os.path.exists('./results/privacy'):
        os.mkdir('./results/privacy')
    pivotSemSim.to_csv('./results/privacy/pivotSemSim.csv')
    pivotLexSim.to_csv('./results/privacy/pivotLexSim.csv')


