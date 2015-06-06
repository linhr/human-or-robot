import numpy as np
import pandas as pd

from utils import *

def get_bidders_train(labels=False):
    df = pd.read_csv(data_file('train.csv'), index_col='bidder_id')
    if not labels:
        df.drop('outcome', axis=1, inplace=True)
    return df

def get_bidders_test(labels=False):
    df = pd.read_csv(data_file('test.csv'), index_col='bidder_id')
    if labels:
        df['outcome'] = pd.Series(np.nan, index=df.index)
    return df

def get_bidders(labels=False):
    train = get_bidders_train(labels=labels)
    test = get_bidders_test(labels=labels)
    return pd.concat([train, test])

def get_bot_bidders():
    train = get_bidders_train(labels=True)
    train.reset_index(level=0, inplace=True)
    bots = train[train['outcome'] == 1]['bidder_id']
    bots.index = np.arange(0, len(bots))
    return bots
