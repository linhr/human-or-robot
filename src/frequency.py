import os

import pandas as pd

from utils import *

@use_bids_data
@use_workspace_directory('frequency')
def count_frequency(conn, column):
    sql = 'SELECT {0}, COUNT(*) AS n FROM bids GROUP BY {0}'.format(column)
    df = pd.read_sql(sql, conn).sort('n')
    output_path = workspace_file('frequency/{0}.csv'.format(column))
    df.to_csv(output_path, header=False, index=False)

if __name__ == '__main__':
    columns = ('auction', 'bidder_id', 'merchandise', 'device', 'country', 'ip', 'url')
    for column in columns:
        count_frequency(column)
