from __future__ import with_statement

import os
import os.path
import errno
import functools
import sqlite3

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
