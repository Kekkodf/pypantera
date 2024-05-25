import pandas as pd
import numpy as np
import faiss
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
from pypantera.src.utils.helper import createLogger
from typing import List
import ir_datasets
from ir_measures import R, calc_aggregate, nDCG, P
import os
from tqdm import tqdm
tqdm.pandas()

IRS = SentenceTransformer("facebook/contriever-msmarco")

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


class AbsractMemmapEncoding:

    def __init__(self, datapath, mappingpath, embedding_size=768, index_name="id", sep=","):
        self.data = np.memmap(datapath, dtype=np.float32, mode="r").reshape(-1, embedding_size)
        self.mapping = pd.read_csv(mappingpath, dtype={index_name: str}, sep=sep).set_index(index_name)

        self.shape = self.get_shape()

    def get_position(self, idx):
        return self.mapping.loc[idx]

    def get_inverse_position(self, offset):
        return self.mapping.loc[self.mapping["offset"]==offset].index[0]


    def get_encoding(self, idx):
        return self.data[self.mapping.loc[idx, "offset"]]

    def get_centroid(self):
        if not hasattr(self, "centroid"):
            self.centroid = np.mean(self.data, axis=0)
        return self.centroid

    def normalize_data(self):
        if not hasattr(self, "normalized_data"):
            self.normalized_data = normalize(self.data)

    def get_normalized_encoding(self, idx):
        self.normalize_data()
        return self.normalized_data[self.mapping.loc[idx, "offset"]]

    def get_data(self, normalized=False):
        if normalized:
            self.normalize_data()
            return self.normalized_data
        else:
            return self.data

    def get_shape(self):

        return self.data.shape


class MemmapCorpusEncoding(AbsractMemmapEncoding):
    def __init__(self, datapath, mappingpath, embedding_size=768):
        super().__init__(datapath, mappingpath, embedding_size, index_name="doc_id")


class MemmapQueriesEncoding(AbsractMemmapEncoding):
    def __init__(self, datapath, mappingpath, embedding_size=768):
        super().__init__(datapath, mappingpath, embedding_size, index_name="qid", sep="\t")
        self.data = self.data[self.mapping.offset, :]
        self.mapping.offset = np.arange(len(self.mapping.index))



def searchIRS(id:pd.Series, queries:pd.Series, indexPath:str, pathMapper:str, k:int) -> pd.DataFrame:
    #read the index for the retrieval
    index:object = faiss.read_index(indexPath)
    #read the mapper
    mapper:list = list(map(lambda x: x.strip(), open(pathMapper, "r").readlines()))
    #encode the queries
    embsQueries:np.array = IRS.encode(queries.tolist())
    #do the searching
    innerProducts, indices = index.search(embsQueries, k) #:np.array, List[int]    
    #create the run
    out:list = []
    for i in range(len(queries)):
        for j in range(k):
            out.append([id[i], mapper[indices[i, j]], j, innerProducts[i, j]])
    run:pd.DataFrame = pd.DataFrame(out, columns=['qid', 'doc_id', 'rank', 'score'])
    return run

def rerankerIRS(run:pd.DataFrame, df:pd.DataFrame, MemmapCorpus:MemmapCorpusEncoding) -> pd.DataFrame:
    def dot_product_matrix(x, y):
        x_expanded = x[:, np.newaxis, :]  # Shape: [n, 1, d]
        y_expanded = y[np.newaxis, :, :]  # Shape: [1, m, d]
        product = np.sum(x_expanded * y_expanded, axis=-1)  # Shape: [n, m]
        return product
    qid = run['qid'].unique()
    for id in qid:
        #get the original query
        og_query:pd.Series = df[df['id'] == id]['text'].values[0]
        embedding_og_query:np.array = IRS.encode(og_query).reshape(1, -1)
        docs_matrics:np.array = MemmapCorpus.get_encoding(list(run[run['qid'] == id]['doc_id']))
        scores:np.array = dot_product_matrix(embedding_og_query, docs_matrics)
        run.loc[run['qid'] == id, 'score'] = np.squeeze(scores)
        #sort the run and reindex the rank
        run = run.sort_values(by=['qid', 'score'], ascending=[True, False])
        run = run.reset_index(drop=True)
        run['rank'] = run.index + 1
        run['Q0'] = 'Q0'
        run['system'] = 'contriever'
        run = run[['qid', 'Q0', 'doc_id', 'rank', 'score', 'system']]
    return run
    ...

def computeRecall(reranked:pd.DataFrame) -> pd.DataFrame:
    #rename qid column to query_id
    reranked = reranked.rename(columns={'qid': 'query_id'})
    #cast score to float
    reranked['score'] = reranked['score'].astype(float)
    #cast rank to int
    reranked['rank'] = reranked['rank'].astype(int)
    #cast doc_id to str
    reranked['doc_id'] = reranked['doc_id'].astype(str)
    #cast query_id to str
    reranked['query_id'] = reranked['query_id'].astype(str)
    #compute the recall
    dataset = ir_datasets.load('msmarco-passage/trec-dl-2019/judged')
    qrels = dataset.qrels_iter()
    qrels = pd.DataFrame(qrels, columns=["query_id", "doc_id", "relevance", "iteration"])
    qrels.query_id = qrels.query_id.astype(str)
    qrels.doc_id = qrels.doc_id.astype(str)
    
    #compute the recall
    res = pd.DataFrame([calc_aggregate([R@100, nDCG@10, P@5], qrels, reranked)])
    #create the output as the single value of the recall
    out = pd.DataFrame(res.iloc[0]).T
    return out

    ...

def pipeline(mech:str, epsilons:List[float]) -> pd.DataFrame:
    #print(f'Processing mechanism: {mech}')
    k:int = 100
    indexPath:str = '../../../../ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/INDEXES/faiss/msmarco-passages/contriever/contriever.faiss'
    pathMapper:str = '../../../../ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/INDEXES/faiss/msmarco-passages/contriever/contriever.map'
    memmapcorpus = MemmapCorpusEncoding("/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/INDEXES/memmap/msmarco-passages/contriever/contriever.dat", "/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/INDEXES/memmap/msmarco-passages/contriever/contriever_map.csv")
    scores:List[float] = []
    for e in epsilons:
        #print(f'Processing epsilon: {e}')
        df:pd.DataFrame = pd.read_csv(f'./results/retrieval/{mech}/obfuscatedText_{mech}_{e}.csv', sep=',', header=0)
        df['obfuscatedText'] = df['obfuscatedText'].astype(str)
        df['text'] = df['text'].astype(str)
        #search the obfuscated text
        logger.info(f'Starting the retrieval process for mechanism: {mech} and epsilon: {e}')
        run:pd.DataFrame = searchIRS(id=df['id'], queries=df['obfuscatedText'], indexPath=indexPath, pathMapper=pathMapper, k=k)
        #rerank the run
        logger.info(f'Starting the reranking process for mechanism: {mech} and epsilon: {e}')
        reranked:pd.DataFrame = rerankerIRS(run, df, memmapcorpus)
        #compute the evaluation of the reranked recall
        logger.info(f'Starting the evaluation process for mechanism: {mech} and epsilon: {e}')
        scoring:pd.DataFrame = computeRecall(reranked)
        #save the results
        logger.info(f'Starting the saving process for mechanism: {mech} and epsilon: {e}')
        scores.append((scoring, e))
    #save the results
    logger.info(f'Saving the results for mechanism: {mech}')
    scores = pd.DataFrame(scores, columns=['scores', 'epsilon'])
    scores['mechanism'] = mech
    return scores

if __name__ == '__main__':
    logger = createLogger()
    mechanisms:List[str] = ['CMP', 'Mahalanobis', 'VickreyCMP', 'VickreyMhl', 'SanText', 'CusText', 'TEM']
    epsilons:List[float] = [1, 5, 10, 12.5, 15, 17.5, 20, 50]
    #creates iterables of mechanisms and epsilons
    mechanisms_epsilons:List[tuple] = [(mech, epsilons) for mech in mechanisms]

    #use multiprocessing
    logger.info('Starting the retrieval pipeline')
    logger.info('Using multiprocessing, number of processes: {}'.format(mp.cpu_count()))
    logger.info('Processing mechanisms: {}'.format(mechanisms))
    logger.info('Processing epsilons: {}'.format(epsilons))
    results:list = []
    for mech in mechanisms:
        temp: object = pipeline(mech, epsilons)
        results.append(temp)
    results = pd.concat(results)
    results.pivot(index='mechanism', columns='epsilon', values='scores').to_csv('./results/retrieval/results.csv', sep=',', header=True, index=True)
