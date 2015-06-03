import os.path
import functools

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

import frequency
import graphs
import timestamp
from utils import *

def feature_fullname(name, prefix=None):
    if not prefix:
        return name
    if not isinstance(prefix, basestring):
        prefix = '__'.join(tuple(prefix))
    return prefix + '__' + name

def feature_file(fullname):
    path = os.path.join('features', fullname.replace('__', os.sep) + '.csv')
    return workspace_file(path)

def save_features(names=(), prefix=''):
    if isinstance(names, basestring):
        names = (names,)
    def wrapper_outer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for name in names:
                path = feature_file(feature_fullname(name, prefix))
                try_makedirs(os.path.dirname(path))
                features = func(name, *args, **kwargs)
                features.to_csv(path, index=True, header=True)
        return wrapper
    return wrapper_outer

class BidderFeature(TransformerMixin):
    def __init__(self, attribute):
        self.attribute = attribute

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tokens = X[self.attribute].apply(lambda s: tuple(ord(x) for x in s))
        return tokens.apply(pd.Series).values

    def get_params(self, deep=True):
        return {'attribute': self.attribute}

class PrecomputedFeature(TransformerMixin):
    def __init__(self, name, default=None, limit=None):
        self.name = name
        self.default = default
        self.limit = limit
        self._load_features()

    def _load_features(self):
        path = feature_file(self.name)
        self.features = pd.read_csv(path, index_col='bidder_id')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = self.features.reindex(X.index).values
        if self.default is not None:
            features[np.isnan(features)] = self.default
        if self.limit is not None:
            features = features[:, :self.limit]
        return features

    def get_params(self, deep=True):
        return {'name': self.name, 'default': self.default, 'limit': self.limit}

@save_features(('merchandise', 'device', 'country', 'ip', 'url'), 'per_auction_freq')
def save_per_auction_freq(name, size=100):
    return frequency.bidder_per_auction_freq(name, auction_count=size)

@save_features(('auction', 'merchandise', 'device', 'country', 'ip', 'url'), 'graph_svd')
def save_graph_svd(name, size=100):
    return graphs.bidder_graph_svd(name, k=size)

@save_features(('auction', 'merchandise', 'device', 'country', 'ip', 'url'), 'cooccurrence_eigen')
def save_cooccurrence_eigen(name, size=100):
    return graphs.bidder_cooccurrence_eigen(name, k=size)

@save_features('response_time_stats', '')
def save_response_time_stats(name):
    return timestamp.get_response_time_statistics()

@save_features('interarrival_time_stats', '')
def save_interarrival_time_stats(name):
    return timestamp.get_interarrival_time_statistics()

if __name__ == '__main__':
    save_per_auction_freq()
    save_graph_svd()
    save_cooccurrence_eigen()
    save_response_time_stats()
    save_interarrival_time_stats()
