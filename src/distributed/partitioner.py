"""Data partitioning utilities for distributed SDG simulation."""

import numpy as np
import pandas as pd


def horizontal_partition(df, n_holders, seed=None):
    """Split rows randomly among n_holders.

    Args:
        df: pandas DataFrame
        n_holders: number of data holders
        seed: random seed

    Returns:
        list of DataFrames, one per holder
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(df))
    splits = np.array_split(indices, n_holders)
    return [df.iloc[s].reset_index(drop=True) for s in splits]


def vertical_partition(df, n_holders, seed=None):
    """Split columns among n_holders. Each partition retains all rows.

    Args:
        df: pandas DataFrame
        n_holders: number of data holders
        seed: random seed

    Returns:
        list of DataFrames (each has all rows, subset of columns)
        Also returns column_map: dict mapping column -> holder_index
    """
    rng = np.random.RandomState(seed)
    columns = list(df.columns)
    rng.shuffle(columns)
    col_splits = np.array_split(columns, n_holders)

    partitions = [df[list(cols)].copy() for cols in col_splits]
    column_map = {}
    for i, cols in enumerate(col_splits):
        for c in cols:
            column_map[c] = i

    return partitions, column_map
