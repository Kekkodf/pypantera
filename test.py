#from pypantera.src.santext import SanText
#from pypantera.src.wbb import WBB
#from pypantera.src.tem import TEM
import time
import numpy as np
import pandas as pd
import os
from typing import List
import argparse
from pypantera.src.cmp import CMP
from pypantera.src.mahalanobis import Mahalanobis
from pypantera.src.vickrey import Vickrey

def main() -> None:
    t0: time = time.time()
    #read the embeddings file from site/embPath.txt
    #with open('/ssd2/data/defaverifr/DATA/embeddings/glove/glove.6B.300d.txt', 'r') as f:
    #    embPath: str = f.readline().strip()
    #print(f"Embeddings file path: {embPath}")
    embPath: str = '/ssd2/data/defaverifr/DATA/embeddings/glove/glove.6B.300d.txt'

    #initialization of the mechanisms
    #mech1: Mechanism = Mechanism({'embPath': embPath, 'epsilon': 40})
    text = 'what is the capitol of france'
    #mech1: CMP = CMP({'embPath': embPath, 'epsilon':4})
    #obfuscatedText = mech1.obfuscateText(text, 1)
    #print(f"Obfuscated text: {obfuscatedText}")
    #mech1: Mahalanobis = Mahalanobis({'embPath': embPath, 'epsilon': 0.1, 'lambda': 1})
    #obfuscatedText = mech1.obfuscateText(text, 1)
    mech1: Vickrey
    #print(f"Obfuscated text: {obfuscatedText}")
    #mech1: Vickrey.CMP = Vickrey.CMP({'embPath': embPath, 'epsilon': 0.1, 'lambda': 1, 't': 0.5})
    #mech2: Vickrey.Mhl = Vickrey.Mhl({'embPath': embPath, 'epsilon': 0.1, 'lambda': 1, 't': 0.5})
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