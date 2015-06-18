import logging

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from bidder import *
from features import *
from utils import *

class PipelineLogger(TransformerMixin):
    logger = logging.getLogger('prediction')

    def fit(self, X, y=None):
        n, m = X.shape
        p = np.sum(y == 1.0)
        self.logger.info('fitting %d samples (%d positive) with %d features', n, p, m)
        return self

    def transform(self, X):
        n, _ = X.shape
        self.logger.info('transforming %d samples', n)
        return X

    def get_params(self, deep=True):
        return {}

def create_pipeline():
    precomputed = [
        ('', ('interarrival_time_stats',), None, None),
        ('', ('response_time_stats',), None, None),
        ('', ('interarrival_steps_stats',), None, None),
        ('', ('bid_amounts_stats',), None, None),
        ('per_auction_freq', ('merchandise', 'device', 'country', 'ip', 'url'), 0., None),
        ('graph_svd', ('auction', 'merchandise', 'device', 'country', 'ip', 'url'), None, None),
        ('cooccurrence_eigen', ('auction', 'merchandise', 'device', 'country', 'ip', 'url'), None, None),
        ('attribute_weight_stats', ('auction', 'device', 'country', 'ip', 'url'), None, None),
    ]
    for rate in ('10s', '30s', '1min', '10min', '30min', '1h', '6h', '12h', '1d'):
        bid_count_series = 'bid_count_series_stats_{0}'.format(rate)
        unique_count_series = 'unique_count_series_stats_{0}'.format(rate)
        series_crosscorr = 'series_crosscorr_{0}'.format(rate)
        precomputed += [
            ('', (bid_count_series,), None, None),
            (unique_count_series, ('auction', 'device', 'country', 'ip', 'url'), None, None),
            ('', (series_crosscorr,), None, None),
        ]

    features = []
    for prefix, names, default, limit in precomputed:
        for name in names:
            fullname = feature_fullname(name, prefix)
            features.append((fullname, PrecomputedFeature(fullname, default=default, limit=limit)))

    pipeline = Pipeline([
        ('features', FeatureUnion(features)),
        ('imputer', Imputer(missing_values='NaN', strategy='mean', axis=0)),
        ('logger', PipelineLogger()),
        ('classifier', RandomForestClassifier(n_estimators=200, max_features='log2')),
    ])
    return pipeline

def get_training_data():
    train = get_bidders_train(labels=True)
    labels = train['outcome'].values
    return train.drop('outcome', axis=1), labels

def get_testing_data():
    return get_bidders_test(labels=False)

def predict():
    pipeline = create_pipeline()
    train, labels = get_training_data()
    pipeline.fit(train, labels)
    test = get_testing_data()
    prediction = pipeline.predict_proba(test)
    (idx,), = np.where(pipeline.classes_ == 1.)
    return pd.Series(prediction[:, idx], index=test.index, name='prediction')

def cross_validation(k=10):
    pipeline = create_pipeline()
    train, labels = get_training_data()
    scores = cross_val_score(pipeline, train, labels, scoring='roc_auc', cv=k)
    return scores

if __name__ == '__main__':
    prediction = predict()
    prediction.to_csv(workspace_file('submission'), index=True, header=True)
