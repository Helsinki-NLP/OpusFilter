"""Create a curriculum learning schedule for a dataset"""

import logging

from sklearn import  preprocessing
from sklearn.cluster import KMeans
import numpy as np

from . import CLEAN_LOW
from . import filters as filtermodule
from .classifier import load_dataframe

logger = logging.getLogger(__name__)

class BabyStep:
    """Cluster segments by filter scores

    Train k-means clustering and split training data into buckets.

    """

    def __init__(self, sample_score_file, k=5, output_file=None, workdir=None):
        self.df = load_dataframe(sample_score_file)
        self.k = k
        self.output_file = output_file
        self.workdir = workdir
        self.filters = {}
        for name in self.df.columns:
            first_part = name.split('.')[0]
            filter_cls = getattr(filtermodule, first_part)
            self.filters[name] = filter_cls
        self.scaler = preprocessing.StandardScaler()
        self.standard_data = self.scaler.fit_transform(self.df)

        logger.info('Training KMeans with %s clusters', self.k)
        self.kmeans = KMeans(n_clusters=self.k, random_state=0, init='k-means++', n_init=1)
        self.kmeans.fit(self.standard_data)

        # The buckets are labeled from cleanest to noisiest with labels from 0 to k 
        self.clean_order = [np.where(np.argsort(np.mean(self.kmeans.cluster_centers_, axis=1))==i)[0][0] for i in range(k)]

    def classify(self, score_file):
        self.df = load_dataframe(score_file)
        self.filters = {}
        for name in self.df.columns:
            first_part = name.split('.')[0]
            filter_cls = getattr(filtermodule, first_part)
            self.filters[name] = filter_cls
        self.scaler = preprocessing.StandardScaler()
        self.standard_data = self.scaler.fit_transform(self.df)

        logger.info(f'Labeling {self.k} buckets with labels 0-{self.k-1}')
        for label in map(lambda x: self.clean_order[x], self.kmeans.predict(self.standard_data)):
            self.output_file.write(str(label)+'\n')
