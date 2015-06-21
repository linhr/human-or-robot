import os

import pandas as pd

from bidder import *
from utils import *

@use_bids_data
@cacheable_data_frame('frequencies/{column}.csv', '{column}')
def attribute_freq(conn, column):
    sql = 'SELECT {0}, COUNT(*) AS n FROM bids GROUP BY {0}'.format(column)
    counts = pd.read_sql(sql, conn, index_col=column).sort('n')
    return counts

def get_popular_auctions(count=None):
    auctions = attribute_freq('auction')
    if count is not None:
        auctions = auctions[-count:]
    return list(auctions.index.values)

@use_bids_data
def bidder_per_auction_freq(conn, column, auction_count=100):
    bidders = get_bidders()
    auctions = get_popular_auctions(auction_count)
    df = pd.DataFrame(data=None, index=bidders.index)
    for auction in auctions:
        sql = ("SELECT bidder_id, COUNT(DISTINCT {0}) as n FROM bids "
            "WHERE auction = '{1}' GROUP BY bidder_id").format(column, auction)
        df[auction] = pd.read_sql(sql, conn, index_col='bidder_id')
    return df

if __name__ == '__main__':
    columns = ('auction', 'bidder_id', 'merchandise', 'device', 'country', 'ip', 'url')
    for column in columns:
        attribute_freq(column)
