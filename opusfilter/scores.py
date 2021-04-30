"""Corpus filtering"""

# pylint: disable=invalid-name, line-too-long, bad-whitespace, too-many-arguments


import logging
# import sys

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from pandas.plotting import scatter_matrix

from tqdm import tqdm

from opustools.util import file_open


def load_scores(fname, includefile=None):
    """ Load score file """

    with file_open(fname, 'r') as fobj:
        df = json_normalize([json.loads(line) for line in tqdm(fobj)])
    if includefile:
        included = set(l.strip() for l in includefile.readlines())
        for col in df.columns:
            if col not in included:
                logging.info('Excluding column %s', col)
                df.drop(col, inplace=True, axis=1)
    return df


def drop_constant_columns(df):
    """ Drop constant columns """

    dropcols = []

    for col in df.columns:
        if len(df[col].unique()) == 1:
            logging.info('Ignoring column %s with constant value', col)
            dropcols.append(col)

    df2 = df.drop(dropcols, axis=1)
    return df2


def scatter(df, figsize=(9,9)):
    """ Build scatter matrix """

    score_columns = list(df.columns)

    subplots = scatter_matrix(df, alpha=0.2, figsize=figsize, diagonal='kde')
    # Long score labels get easily overlapped, decrease size and
    # rotate to help a bit
    for ridx, row in enumerate(subplots):
        for cidx, ax in enumerate(row):
            if ridx == len(df.columns) - 1:
                ax.set_xlabel(score_columns[cidx], rotation=30, fontsize=7)
            if cidx == 0:
                ax.set_ylabel(score_columns[ridx], rotation=30, fontsize=7)


def correlations(df, figsize=(6,6)):
    """ Correlation matrix """

    score_columns = list(df.columns)

    _corr = df.corr()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cmap = ax.matshow(_corr)
    ticks = np.arange(len(score_columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels(score_columns, rotation=90, fontsize=7)
    ax.set_yticklabels(score_columns, fontsize=7)
    cb = fig.colorbar(cmap, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Score correlations')
    fig.tight_layout()


def hist(df, bins=50, log=False, figsize=(9,9), layout=None):
    """ Histogram """

    matplotlib.rcParams.update({'font.size': 7})
    df.hist(bins=bins, log=log, figsize=figsize, layout=layout)
    plt.tight_layout()


def values(df, figsize=(11,4)):
    """ Values """

    for idx, col in enumerate(df.columns):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.plot(df.index, df[col])
        ax.set_title('{}'.format(col))
        fig.tight_layout()


def misfits(df, col, source, target, asc=True, num=10):
    """ Show extreme values """

    df_sorted = df.sort_values(by=[col], ascending=asc)
    #df_sorted.dtypes

    subset = df_sorted.head(num)
    src = []
    trg = []
    vals = []
    for idx in subset.index:
        src.append(source[idx])
        trg.append(target[idx])
        vals.append(df[col].iloc[idx])
        # print(sorted[col].iloc[idx])

    newdf = pd.DataFrame(np.array([src, trg, vals]).T, columns=['source', 'target', 'value'])
    newdf.index = subset.index
    return newdf
#    return src, trg, list(subset.index), vals
