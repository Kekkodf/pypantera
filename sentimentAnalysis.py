import pandas as pd
from typing import List
from transformers import pipeline, AutoModelForSequenceClassification

import os
from tqdm import tqdm
tqdm.pandas() 

if __name__ == '__main__':
    mechanisms:List[str] = ['CMP', 'Mahalanobis', 'VickreyCMP', 'VickreyMhl', 'SanText', 'CusText', 'TEM']
    epsilons:List[float] = [1, 5, 10, 12.5, 15, 17.5, 20, 50]

    #initialize the sentiment analysis pipeline
    MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    sentiment_analysis = pipeline('sentiment-analysis', model=model, tokenizer=MODEL)

    df_sent = pd.DataFrame(columns=['id','obfuscatedText', 'trueLabel', 'predictedLabel', 'mechanism', 'epsilon'])
    results = []
    for mech in mechanisms:
        print(f'Processing mechanism: {mech}')
        for e in epsilons:
            print(f'Processing epsilon: {e}')
            df = pd.read_csv(f'./results/sentimentAnalysis/{mech}/obfuscatedText_{mech}_{e}.csv', sep=',', header=0)
            df_sent['trueLabel'] = df['sentiment']
            df['obfuscatedText'] = df['obfuscatedText'].astype(str)
            df_sent['predictedLabel'] = df['obfuscatedText'].progress_apply(lambda x: sentiment_analysis(x)[0]['label'])
            df_sent['id'] = df['id']
            df_sent['mechanism'] = mech
            df_sent['epsilon'] = e
            #calculate the accuracy
            accuracy = (df_sent['trueLabel'].str.lower() == df_sent['predictedLabel'].str.lower()).mean()
            tuple_mech = (mech, e, accuracy)
            results.append(tuple_mech)
    list_of_results = [{'mechanism': mech, 'epsilon': e, 'accuracy': accuracy} for mech, e, accuracy in results]
    results_df = pd.DataFrame(list_of_results, columns=['mechanism', 'epsilon', 'accuracy'])
    #create a pivot table with the results
    pivotAcc = results_df.pivot(index='mechanism', columns='epsilon', values='accuracy')
    #save the pivot table to a csv file
    if not os.path.exists('./results/sentimentAnalysis/evaluation'):
        os.mkdir('./results/sentimentAnalysis/evaluation')
    pivotAcc.to_csv('./results/sentimentAnalysis/evaluation/sentimentAnalysisAccuracy.csv')

    #if you want also to get the accuracy on the original texts, 
    #just perform the same operations on the original text (it is irrelevant from which dataset you get the text, just be sure to get the text column and the sentiment column )
    #df = pd.read_csv('./results/sentimentAnalysis/CMP/obfuscatedText_CMP_1.csv', sep=',', header=0)
    #df['text'] = df['text'].astype(str)
    #df['trueLabel'] = df['sentiment']
    #df['predictedLabel'] = df['text'].progress_apply(lambda x: sentiment_analysis(x)[0]['label'])
    #accuracy = (df['trueLabel'].str.lower() == df['predictedLabel'].str.lower()).mean()
    #print(f'Accuracy on original text: {accuracy}')