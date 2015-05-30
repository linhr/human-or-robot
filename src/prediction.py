import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC

from bidder import *
from features import *

class PipelineLogger(TransformerMixin):
    def fit(self, X, y=None):
        n, m = X.shape
        p = np.sum(y == 1.0)
        print 'fitting %d samples (%d positive) with %d features' % (n, p, m)
        return self

    def transform(self, X):
        n, _ = X.shape
        print 'transforming %d samples' % n
        return X

def create_pipeline():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('account', BidderFeature('payment_account')),
            ('address', BidderFeature('address')),
        ])),
        ('logger', PipelineLogger()),
        ('classifier', SVC(probability=True)),
    ])
    return pipeline

def _get_training_data():
    train = get_bidders_train(labels=True)
    labels = train['outcome'].values
    return train.drop('outcome', axis=1), labels

def _get_testing_data():
    return get_bidders_test(labels=False)

def predict():
    pipeline = create_pipeline()
    train, labels = _get_training_data()
    pipeline.fit(train, labels)
    test = _get_testing_data()
    prediction = pipeline.predict_proba(test)
    (idx,), = np.where(pipeline.classes_ == 1.)
    return pd.Series(prediction[:, idx], index=test.index, name='prediction')

def cross_validation(k=10):
    pipeline = create_pipeline()
    train, labels = _get_training_data()
    scores = cross_val_score(pipeline, train, labels, scoring='roc_auc', cv=k)
    return scores

if __name__ == '__main__':
    prediction = predict()
    prediction.to_csv(workspace_file('submission'), index=True, header=True)
