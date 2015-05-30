import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

class BidderFeature(TransformerMixin):
    def __init__(self, attribute):
        self.attribute = attribute

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tokens = X[self.attribute].apply(lambda s: tuple(ord(x) for x in s))
        return tokens.apply(pd.Series).values
