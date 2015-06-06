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

@use_bids_data
@cacheable_data_frame('series/unique_count_{rate}/{column}.pickle.gz', 'time')
def unique_count(conn, column, rate='1min'):
    def sampler(x):
        u = x.groupby('bidder_id').apply(lambda x: x[column].nunique())
        return u if len(u) > 0 else pd.Series()

    sql = 'SELECT bidder_id, {0}, time FROM bids'.format(column)
    df = pd.read_sql(sql, conn)
    df = _build_time_index(df)
    df = df.groupby(pd.TimeGrouper(rate)).apply(sampler)
    df = df.unstack()
    return df
