import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse.linalg

import frequency
from utils import *

@use_bids_data
@cacheable_data_frame('graphs/{column}.csv.gz', ['bidder_id', '{column}'])
def bidder_graph(conn, column):
    sql = ("SELECT bidder_id, {0}, COUNT(*) as weight FROM bids "
        "GROUP BY bidder_id, {0}").format(column)
    df = pd.read_sql(sql, conn, index_col=['bidder_id', column])
    return df

def bidder_graph_svd(column, k=100):
    # `edges` is a DataFrame with a multi-index. The first-level index is
    # `bidder_id` for source nodes, and the second-level index is for destination nodes.
    edges = bidder_graph(column)
    matrix, rows, cols = edges['weight'].to_sparse().to_coo()
    _, c = matrix.shape
    u, _, _ = sp.sparse.linalg.svds(matrix.astype(float), k=min(k, c-1))
    df = pd.DataFrame(data=u, index=rows)
    df.index.name = 'bidder_id'
    return df

@cacheable_data_frame('cooccurrence/{column}.pickle.gz', ['bidder_id_x', 'bidder_id_y'])
def cooccurrence_graph(column):
    df = bidder_graph(column).reset_index().drop('weight', axis=1)
    df = pd.merge(df, df, on=column)
    df = df.groupby(['bidder_id_x', 'bidder_id_y']).aggregate('count')
    df.columns = ['weight']
    return df

def _cooccurrence_adjacency_matrix(column):
    df = cooccurrence_graph(column).reset_index()
    df = df.pivot(index='bidder_id_x', columns='bidder_id_y', values='weight')
    df = df[df.index.tolist()] # ensure the order of columns
    matrix = df.values
    matrix[np.isnan(matrix)] = 0.
    matrix = sp.sparse.csr_matrix(matrix)
    return matrix, df.index

def bidder_cooccurrence_eigen(column, k=100):
    matrix, index = _cooccurrence_adjacency_matrix(column)
    n, _ = matrix.shape
    _, v = sp.sparse.linalg.eigs(matrix, k=min(k, n-2))
    df = pd.DataFrame(data=v.real, index=index)
    df.index.name = 'bidder_id'
    return df

def bid_attribute_weight_statistics(column):
    quantiles = np.array([25, 50, 75])
    names = ['count', 'min', 'max', 'mean', 'std', 'kurtosis']
    names += ['percentile_' + str(x) for x in quantiles]
    def stats(x):
        s = np.array([len(x), x.min(), x.max(), x.mean(), x.std(), x.kurtosis()])
        q = np.percentile(x, quantiles)
        return pd.Series(np.hstack([s, q]))

    graph = bidder_graph(column)
    graph.reset_index(level=1, inplace=True)
    graph = graph.drop('weight', axis=1).dropna()
    freq = frequency.attribute_freq(column)
    graph = graph.join(freq, on=column, how='inner').drop(column, axis=1)
    result = graph['n'].groupby(level=0).apply(stats).unstack()
    result.columns = names
    result.index.name = 'bidder_id'
    return result
