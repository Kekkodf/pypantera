import numpy as np
import pandas as pd
from typing import List

def sentenceSimilarity(model:object, text:str, obfuscatedText:str) -> float:
    '''
    sentenceSimilarity: This function calculates the contextual similarity between two sentences using a pre-trained model.

    input:
    model: object - A pre-trained model to calculate the similarity between sentences.
    text: str - The original sentence.
    obfuscatedText: str - The obfuscated sentence.

    output:
    float - The similarity value between the two sentences.

    Example:
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    text = 'The quick brown fox jumps over the lazy turtle.'
    obfuscatedText = 'The fast brown dog jumps over the lazy cat.'
    similarity = sentenceSimilarity(model, text, obfuscatedText)
    print(similarity)
    '''
    textEmbedding = model.encode(text)
    obfuscatedTextEmbedding = model.encode(obfuscatedText)
    similarity = np.dot(textEmbedding, obfuscatedTextEmbedding) / (np.linalg.norm(textEmbedding) * np.linalg.norm(obfuscatedTextEmbedding))
    return similarity
    ...

def lexicalSimilarity(text:str, obfuscatedText:str) -> float:
    '''
    lexicalSimilarity: This function calculates the lexical similarity between two sentences 
                       by implementing the Jaccard similarity function.

    input:
    text: str - The original sentence.
    obfuscatedText: str - The obfuscated sentence.

    output:
    float - The similarity value between the two sentences.

    Example:
    text = 'The quick brown fox jumps over the lazy turtle.'
    obfuscatedText = 'The fast brown dog jumps over the lazy cat.'
    similarity = lexicalSimilarity(text, obfuscatedText)
    print(similarity)    
    '''
    textTokens = set(text.split())
    obfuscatedTextTokens = set(obfuscatedText.split())
    intersection = len(textTokens.intersection(obfuscatedTextTokens))
    union = len(textTokens.union(obfuscatedTextTokens))
    similarity = intersection / union
    return similarity