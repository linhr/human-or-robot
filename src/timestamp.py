import numpy as np
import pandas as pd

import frequency
from utils import *

@use_bids_data
def get_timestamps(conn, auction, merchandise=None):
    cond = ("auction = '{0}'".format(auction),)
    if merchandise:
        cond += ("merchandise = '{0}'".format(merchandise),)
    sql = 'SELECT bidder_id, time FROM bids WHERE {0}'.format(' AND '.join(cond))
    df = pd.read_sql(sql, conn, index_col='bidder_id')
    return df

def get_response_time(auction, merchandise=None):
    df = get_timestamps(auction, merchandise).sort('time').diff()
    return df

def get_interarrival_time(auction, merchandise=None):
    df = get_timestamps(auction, merchandise)
    df = df.groupby(level='bidder_id').transform(lambda x: x.order().diff())
    return df

def _get_time_statistics(auction_count, data_loader):
    auctions = frequency.get_popular_auctions(auction_count)
    df = pd.DataFrame()
    for auction in auctions:
        df = df.append(data_loader(auction))
    df = df.dropna(how='any')
    grouped = df.groupby(level=0)
    grouped = grouped['time']
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

def get_response_time_statistics(auction_count=100):
    return _get_time_statistics(auction_count, get_response_time)

def get_interarrival_time_statistics(auction_count=100):
    return _get_time_statistics(auction_count, get_interarrival_time)
