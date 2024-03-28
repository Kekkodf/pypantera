from pypanter.src.mechanism import Mechanism
from pypanter.src.mahalanobis import Mahalanobis
import time
import numpy as np
import pandas as pd
import os
from typing import List

def main() -> None:
    t0: time = time.time()
    embPath: str = "/ssd2/data/defaverifr/DATA/embeddings/glove/glove.6B.300d.txt"
    #mech1: Mechanism = Mechanism({'embPath': embPath, 'epsilon': 0.1})
    mech1: Mahalanobis = Mahalanobis({'embPath': embPath, 'epsilon':20, 'lambda': 1})
    df: pd.DataFrame = pd.DataFrame(columns=['text', 'obfuscatedText'])
    df['text'] = ['what is the capitol of france', 'what is the capitol of germany', 'what is the capitol of italy']
    df['obfuscatedText'] = df['text'].apply(lambda x: mech1.obfuscateText(x, 10))
    #in the obfuscated text column we have a list of obfuscated queries, create new lines in the df for each query
    df = df.explode('obfuscatedText')
    df['mechanism'] = mech1.__class__.__name__
    #save the obfuscated text to a csv file
    if not os.path.exists('test/obfuscated'):
        os.makedirs('test/obfuscated')
    df.to_csv(f'test/obfuscated/obfuscatedTestRun_{mech1.__class__.__name__}.csv', index=False)
    print(f"Time taken: {time.time() - t0}")
    

if __name__ == '__main__':
    main()