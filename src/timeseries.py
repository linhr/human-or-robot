import pandas as pd
import numpy as np

from utils import *
import timestamp

def _build_time_index(df):
    stat = timestamp.get_basic_statistics()
    offset = stat['min']['value']
    resolution = stat['resolution']['value']
    df['time'] -= offset
    df['time'] /= resolution
    df.index = pd.to_datetime(df['time'], unit='s')
    df = df.drop('time', axis=1)
    df.sort_index(inplace=True)
    return df

def _get_time_series(df, rate, aggregator):
    def agg(x):
        u = aggregator(x)
        return u if len(u) > 0 else pd.Series()

    df = _build_time_index(df)
    df = df.groupby(pd.TimeGrouper(rate)).apply(agg)
    return df

@use_bids_data
@cacheable_data_frame('series/unique_count_{rate}/{column}.h5', 'time')
def unique_count(conn, column, rate='1min'):
    def aggregator(x):
        return x.groupby('bidder_id').apply(lambda x: x[column].nunique())

    sql = 'SELECT bidder_id, {0}, time FROM bids'.format(column)
    df = pd.read_sql(sql, conn)
    return _get_time_series(df, rate, aggregator).unstack()

@use_bids_data
@cacheable_data_frame('series/bid_count_{rate}.h5', 'time')
def bid_count(conn, rate='1min'):
    def aggregator(x):
        return x.groupby('bidder_id').apply(lambda x: len(x))

    sql = 'SELECT bidder_id, time FROM bids'
    df = pd.read_sql(sql, conn)
    return _get_time_series(df, rate, aggregator).unstack()

def _entropy(series):
    s = series.dropna().values
    return -np.nansum(s * np.log2(s))

def _auto_correlation(series, count):
    count = max(min(count, len(series)-1), 0)
    v = series.values
    return np.array([np.corrcoef(v[:-i], v[i:])[0, 1] for i in xrange(1, count+1)])

def _statistics_extractor(series, autocorr_count=10, named=False):
    stats = np.array([
        series.min(),
        series.max(),
        series.mean(),
        series.std(),
        series.kurtosis(),
        _entropy(series),
    ])
    series.fillna(0, inplace=True)
    autocorr = _auto_correlation(series, autocorr_count)
    stats = pd.Series(np.hstack((stats, autocorr)))
    if named:
        names = ['min', 'max', 'mean', 'std', 'kurtosis', 'entropy']
        for i in xrange(len(autocorr)):
            names.append('autocorr_{0}'.format(i+1))
        stats.index = names
    return stats

def _get_series_statistics(df):
    # each column of the input data frame is a time series for a bidder
    df = df.T
    if len(df) == 0:
        raise ValueError('empty bidder record')
    first = df.iloc[0]
    df = df.apply(_statistics_extractor, axis=1)
    # use one series to determine column names
    df.columns = _statistics_extractor(first, named=True).index.tolist()
    return df

def unique_count_statistics(column, rate='1min'):
    return _get_series_statistics(unique_count(column, rate))

def bid_count_statistics(rate='1min'):
    return _get_series_statistics(bid_count(rate))
