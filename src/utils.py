from __future__ import with_statement

import cPickle as pickle
import os
import os.path
import errno
import functools
import gzip
import inspect
import sqlite3

import pandas as pd

import settings

def data_file(path):
    return os.path.join(settings.DATA_PATH, path)

def workspace_file(path):
    return os.path.join(settings.WORKSPACE_PATH, path)

def try_makedirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass

def use_bids_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with sqlite3.connect(data_file('bids.db')) as conn:
            return func(conn, *args, **kwargs)
    return wrapper

def use_workspace_directory(path):
    def wrapper_outer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try_makedirs(workspace_file(path))
            return func(*args, **kwargs)
        return wrapper
    return wrapper_outer

def cacheable_data_frame(pathfmt, indexfmt):
    def format_index_col(index_col_fmt, *args, **kwargs):
        isstr = lambda x: isinstance(x, basestring)
        formatted = lambda x: x.format(*args, **kwargs)
        if isstr(index_col_fmt):
            return formatted(index_col_fmt)
        elif isinstance(index_col_fmt, (list, tuple)):
            return [formatted(x) if isstr(x) else x for x in index_col_fmt]
        return index_col_fmt

    def save_data_frame(df, full_path):
        try_makedirs(os.path.dirname(full_path))
        path_lower = full_path.lower()
        if path_lower.endswith('.csv'):
            with open(full_path, 'w') as output:
                df.to_csv(output, index=True, header=True)
        elif path_lower.endswith('.csv.gz'):
            with gzip.open(full_path, 'wb') as output:
                df.to_csv(output, index=True, header=True)
        elif path_lower.endswith('.pickle'):
            with open(full_path, 'wb') as output:
                pickle.dump(df, output, protocol=pickle.HIGHEST_PROTOCOL)
        elif path_lower.endswith('.pickle.gz'):
            with gzip.open(full_path, 'wb') as output:
                pickle.dump(df, output, protocol=pickle.HIGHEST_PROTOCOL)
        elif path_lower.endswith('.h5'):
            df.to_hdf(full_path, 'df', mode='w', complevel=1, complib='zlib')
        else:
            raise ValueError('unrecognized format')

    def load_data_frame(full_path, index_col=None):
        path_lower = full_path.lower()
        if path_lower.endswith('.csv') or path_lower.endswith('.csv.gz'):
            # let pandas handle gzipped csv files
            return pd.read_csv(full_path, index_col=index_col)
        elif path_lower.endswith('.pickle'):
            with open(full_path, 'rb') as data:
                return pickle.load(data)
        elif path_lower.endswith('.pickle.gz'):
            with gzip.open(full_path, 'rb') as data:
                return pickle.load(data)
        elif path_lower.endswith('.h5'):
            return pd.read_hdf(full_path, 'df')
        else:
            raise ValueError('unrecognized format')

    def wrapper_outer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fmtargs = inspect.getcallargs(func, *args, **kwargs)
            full_path = workspace_file(pathfmt.format(**fmtargs))
            if os.path.exists(full_path):
                index_col = format_index_col(indexfmt, **fmtargs)
                return load_data_frame(full_path, index_col=index_col)
            result = func(*args, **kwargs)
            save_data_frame(result, full_path)
            return result
        return wrapper
    return wrapper_outer
