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

def _cross_correlation(series_x, series_y, start, end):
    def corrcoef(a, b):
        return np.corrcoef(a, b)[0, 1]

    if len(series_x) != len(series_y):
        raise ValueError('series length disagree')
    length = len(series_x)
    if start < 0:
        raise ValueError('invalid start value')
    end = max(min(end, length), 0)
    u = series_x.values
    v = series_y.values
    corr = [corrcoef(u[0:length-i], v[i:length]) for i in xrange(start, end)]
    return np.array(corr)

def _auto_correlation(series, count):
    return _cross_correlation(series, series, 1, count+1)

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

def _get_crosscorr_2(df1, df2, start, end):
    result = pd.DataFrame()
    for bidder in df1.columns:
        try:
            s0, s1 = df1[bidder], df2[bidder]
        except KeyError:
            continue
        s0.fillna(0, inplace=True)
        s1.fillna(0, inplace=True)
        result[bidder] = pd.Series(_cross_correlation(s0, s1, start, end))
    return result.T

def get_crosscorr(series, start=0, end=3):
    result = pd.DataFrame()
    for name1, loader1 in series.iteritems():
        df1 = loader1()
        for name2, loader2 in series.iteritems():
            if name1 == name2:
                continue
            df2 = loader2()
            crosscorr = _get_crosscorr_2(df1, df2, start, end)
            crosscorr.columns = ['{0}_vs_{1}_{2}'.format(name1, name2, i) \
                for i in xrange(start, start+len(crosscorr.columns))]
            result = pd.concat([result, crosscorr], axis=1)
    result.index.name = 'bidder_id'
    return result
