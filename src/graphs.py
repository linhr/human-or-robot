import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse.linalg

from utils import *

@use_bids_data
@cacheable_data_frame('graphs/{column}.csv.gz', ['bidder_id', '{column}'])
def bidder_graph(conn, column):
    sql = ("SELECT bidder_id, {0}, COUNT(*) as weight FROM bids "
        "GROUP BY bidder_id, {0}").format(column)
    df = pd.read_sql(sql, conn, index_col=['bidder_id', column])
    return df

def bidder_graph_svd(column, k=100):
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
