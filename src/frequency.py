import os

import pandas as pd

from bidder import *
from utils import *

@use_bids_data
def attribute_freq(conn, column):
    sql = 'SELECT {0}, COUNT(*) AS n FROM bids GROUP BY {0}'.format(column)
    counts = pd.read_sql(sql, conn, index_col=column).sort('n')
    return counts['n']

@use_bids_data
def bidder_per_auction_freq(conn, column, auction_count=100):
    bidders = get_bidders()
    auctions = attribute_freq('auction')[-auction_count:]
    df = pd.DataFrame(data=None, index=bidders.index)
    for auction in auctions.index:
        sql = ("SELECT bidder_id, COUNT(DISTINCT {0}) as n FROM bids "
            "WHERE auction = '{1}' GROUP BY bidder_id").format(column, auction)
        df[auction] = pd.read_sql(sql, conn, index_col='bidder_id')
    return df

@use_workspace_directory('attributes')
def save_attribute_freq():
    columns = ('auction', 'bidder_id', 'merchandise', 'device', 'country', 'ip', 'url')
    for column in columns:
        path = workspace_file('attributes/{0}.csv'.format(column))
        attribute_freq(column).to_csv(path, index=True, header=True)

if __name__ == '__main__':
    save_attribute_freq()
