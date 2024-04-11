#from pypantera.src.santext import SanText
#from pypantera.src.wbb import WBB
#from pypantera.src.tem import TEM
import time
import numpy as np
import pandas as pd
import os
from typing import List
import argparse

from pypantera.src.mechanism import Mechanism
from pypantera.src.cmp import CMP
from pypantera.src.mahalanobis import Mahalanobis
from pypantera.src.vickrey import VickreyCMP, VickreyMhl
from pypantera.src.wbb import WBB
from pypantera.src.custext import CusText
#from pypantera.src.santext import SanText
#from pypantera.src.tem import TEM

def main() -> None:
    t0: time = time.time()
    #read the embeddings file from site/embPath.txt
    #with open('/ssd2/data/defaverifr/DATA/embeddings/glove/glove.6B.300d.txt', 'r') as f:
    #    embPath: str = f.readline().strip()
    #print(f"Embeddings file path: {embPath}")
    embPath: str = '/ssd2/data/defaverifr/DATA/embeddings/glove/glove.6B.300d.txt'

    #initialization of the mechanisms
    #mech1: Mechanism = Mechanism({'embPath': embPath, 'epsilon': 40})
    #important notes: The text should be on lowercase and the words should be separated by a space
    texts = ['what is the capitol of france', 'who stole the bread', 'how many times does the cat jump on the table']

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
    mech1: CusText = CusText({'embPath': embPath, 'epsilon': 5, 'k': 4}) 
    for text in texts:
        text = text.lower().split()
        obfuscatedText = mech1.tokenMappingGeneration(text)
        print(f"Obfuscated text: {obfuscatedText}")
    
    

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
    main()