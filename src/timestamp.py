import numpy as np
import pandas as pd

from utils import *

@use_bids_data
def get_auction_timestamps(conn, auction=None):
    sql = "SELECT time FROM bids"
    if auction:
        sql += " WHERE auction = '{0}'".format(auction)
    df = pd.read_sql(sql, conn)
    return df['time'].order()

def get_interarrival_time_distribution(auction=None):
    t = get_auction_timestamps(auction)
    t = t.diff()
    t = t.groupby(t).agg('count')
    return t

@cacheable_data_frame('misc/timestamp_stat.csv', 'key')
def get_basic_statistics():
    t = get_auction_timestamps()
    d = get_interarrival_time_distribution()
    d = d.index.values
    df = pd.DataFrame(data=None, index=['value'])
    df['min'] = t.min()
    df['max'] = t.max()
    df['resolution'] = d[d > 0].min()
    df.index.name = 'key'
    return df

@use_bids_data
def _get_timestamp_groups(conn):
    sql = 'SELECT bidder_id, auction, time FROM bids'
    df = pd.read_sql(sql, conn, index_col='bidder_id')
    return df.groupby('auction')['time']

def _time_delta(timestamps):
    return timestamps.order().diff()

def get_response_time():
    df = _get_timestamp_groups()
    df = df.transform(_time_delta)
    return df.dropna(how='any')

def get_interarrival_time():
    df = _get_timestamp_groups()
    df = df.transform(lambda x: x.groupby(level=0).transform(_time_delta))
    return df.dropna(how='any')

def _steps(timestamps):
    timestamps = timestamps.order()
    timestamps[:] = np.arange(0, len(timestamps))
    return timestamps

def _amounts(steps):
    return steps.groupby((steps.diff() != 1).cumsum()).agg('count')

def get_interarrival_steps():
    df = _get_timestamp_groups()
    df = df.transform(lambda x: _steps(x).groupby(level=0).transform(pd.Series.diff))
    return df.dropna(how='any')

def get_bid_amounts():
    df = _get_timestamp_groups()
    df = df.apply(lambda x: _steps(x).groupby(level=0).apply(_amounts))
    df.index = df.index.droplevel(2).droplevel(0)
    return df

def _get_series_statistics(data_loader, normalize_percentiles=True):
    df = data_loader()
    grouped = df.groupby(level=0)
    stats = grouped.agg({
        'count': np.size,
        'min': np.min,
        'max': np.max,
        'mean': np.mean,
        'std': np.std,
    })
    q = np.arange(0, 100, 10)
    if normalize_percentiles:
        get_percentiles = lambda x: pd.Series(np.percentile(x, q) / np.max(x))
    else:
        get_percentiles = lambda x: pd.Series(np.percentile(x, q))
    percentiles = grouped.apply(get_percentiles)
    percentiles = percentiles.unstack()
    percentiles.columns = ['percentile_' + str(x) for x in q]
    result = pd.concat([stats, percentiles], axis=1, join='inner')
    result.index.name = 'bidder_id'
    return result

def get_response_time_statistics():
    return _get_series_statistics(get_response_time)

def get_interarrival_time_statistics():
    return _get_series_statistics(get_interarrival_time)

def get_interarrival_steps_statistics():
    return _get_series_statistics(get_interarrival_steps, False)

def get_bid_amounts_statistics():
    return _get_series_statistics(get_bid_amounts, False)
