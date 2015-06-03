import numpy as np
import pandas as pd

from utils import *

@use_bids_data
def _get_timestamp_groups(conn):
    sql = 'SELECT bidder_id, auction, time FROM bids'
    df = pd.read_sql(sql, conn, index_col='bidder_id')
    return df.groupby('auction')['time']

def _time_delta(series):
    return series.order().diff()

def get_response_time():
    df = _get_timestamp_groups()
    df = df.transform(_time_delta)
    return df.dropna(how='any')

def get_interarrival_time():
    df = _get_timestamp_groups()
    df = df.transform(lambda x: x.groupby(level='bidder_id').transform(_time_delta))
    return df.dropna(how='any')

def _get_time_statistics(data_loader):
    df = data_loader()
    grouped = df.groupby(level=0)
    stats = grouped.agg({
        'min': np.min,
        'max': np.max,
        'std': np.std,
    })
    # compute "normalized" percentiles
    q = np.arange(0, 100, 10)
    percentiles = grouped.apply(lambda x: pd.Series(np.percentile(x, q) / np.max(x)))
    percentiles = percentiles.unstack()
    percentiles.columns = ['percentile_' + str(x) for x in q]
    result = pd.concat([stats, percentiles], axis=1, join='inner')
    result.index.name = 'bidder_id'
    return result

def get_response_time_statistics():
    return _get_time_statistics(get_response_time)

def get_interarrival_time_statistics():
    return _get_time_statistics(get_interarrival_time)
