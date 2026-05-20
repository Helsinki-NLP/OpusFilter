"""Create a curriculum learning schedule for a dataset"""

import logging
from collections import Counter
from statistics import geometric_mean

from sklearn import  preprocessing
from sklearn.cluster import KMeans
import numpy as np

from . import CLEAN_LOW
from . import filters as filtermodule
from .util import load_dataframe_in_chunks
from .classifier import load_dataframe

logger = logging.getLogger(__name__)

class BabyStep:
    """Cluster segments by filter scores

    Train k-means clustering and split training data into buckets.

    """

    def __init__(self, sample_score_file, data_size, k=10, data_inc=100000, gmean=False, output_file=None, workdir=None, chunksize=500000):
        self.df = load_dataframe(sample_score_file)
        self.k = k + int((data_size - data_inc)/data_inc)
        self.output_file = output_file
        self.workdir = workdir
        self.chunksize = chunksize

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
        logger.info(f'Sample label distribution (clean=0, noisy={self.k-1}): {dict(sorted(Counter(self.kmeans.labels_).items()))}')

        # Low values are clean, high values are noisy
        adjusted_centers = self.kmeans.cluster_centers_ * self.direction_vector
    
        if gmean:
            temp_centers = adjusted_centers + abs(adjusted_centers.min()) + 0.01
            means = [geometric_mean(m) for m in temp_centers]
        else:
            means = np.mean(adjusted_centers, axis=1)

        # The buckets are labeled from cleanest to noisiest with labels from 0 to k 
        self.clean_order = [np.where(np.argsort(means)==i)[0][0] for i in range(self.k)]

    @property
    def direction_vector(self):
        """Direction vector for the features (1 for CLEAN_LOW, -1 for CLEAN_HIGH)"""
        return np.array([1 if self.filters[name].score_direction == CLEAN_LOW else -1
                         for name in self.df.columns])

    def classify(self, score_file):
        logger.info(f'Dividing training data into {self.k} buckets with labels 0-{self.k-1}')
        df_chunks = load_dataframe_in_chunks(score_file, self.chunksize)

        for df in df_chunks:
            self.standard_data = self.scaler.fit_transform(df)

            for label in map(lambda x: self.clean_order[x], self.kmeans.predict(self.standard_data)):
                self.output_file.write(str(label)+'\n')

        logger.info(f'Labels written to {self.output_file.name}')
