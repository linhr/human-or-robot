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
@cacheable_data_frame('series/unique_count_{rate}/{column}.pickle.gz', 'time')
def unique_count(conn, column, rate='1min'):
    def aggregator(x):
        return x.groupby('bidder_id').apply(lambda x: x[column].nunique())

    sql = 'SELECT bidder_id, {0}, time FROM bids'.format(column)
    df = pd.read_sql(sql, conn)
    return _get_time_series(df, rate, aggregator).unstack()

@use_bids_data
@cacheable_data_frame('series/bid_count_{rate}.pickle.gz', 'time')
def bid_count(conn, rate='1min'):
    def aggregator(x):
        return x.groupby('bidder_id').apply(lambda x: len(x))

    sql = 'SELECT bidder_id, time FROM bids'
    df = pd.read_sql(sql, conn)
    return _get_time_series(df, rate, aggregator).unstack()
