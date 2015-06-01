from __future__ import with_statement

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

    def save_csv(df, full_path):
        try_makedirs(os.path.dirname(full_path))
        if full_path.lower().endswith('.gz'):
            with gzip.open(full_path, 'wb') as output:
                df.to_csv(output, index=True, header=True)
        else:
            df.to_csv(full_path, index=True, header=True)

    def wrapper_outer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fmtargs = inspect.getcallargs(func, *args, **kwargs)
            full_path = workspace_file(pathfmt.format(**fmtargs))
            if os.path.exists(full_path):
                index_col = format_index_col(indexfmt, **fmtargs)
                return pd.read_csv(full_path, index_col=index_col)
            result = func(*args, **kwargs)
            save_csv(result, full_path)
            return result
        return wrapper
    return wrapper_outer
