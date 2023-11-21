"""Unsupervised threshold selection for filters"""

from collections import Counter
import logging

import pandas as pd
from sklearn.cluster import KMeans
from sklearn import decomposition, preprocessing, random_projection
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np

from . import CLEAN_LOW
from . import filters as filtermodule
from .classifier import load_dataframe


logger = logging.getLogger(__name__)


class ScoreClusters:
    """Cluster segments by filter scores

    Train k-means clustering and take thresholds based on the noisy cluster center.

    """

    def __init__(self, score_file, k=2, extreme=None):
        self.k = k
        self.df = load_dataframe(score_file)
        self.filters = {}
        for name in self.df.columns:
            first_part = name.split('.')[0]
            filter_cls = getattr(filtermodule, first_part)
            self.filters[name] = filter_cls
        self.scaler = preprocessing.StandardScaler()
        self.standard_data = self.scaler.fit_transform(self.df.mul(self.direction_vector))

        # Find the noisiest and cleanest examples
        if extreme:
            if extreme == 'mean':
                self.standard_data = self.get_extreme_data_mean(self.standard_data)
        #self.standard_data = self.get_extreme_data_euclidean(self.standard_data)

        logger.info('Training KMeans with %s clusters', self.k)
        self.kmeans = KMeans(n_clusters=self.k, random_state=0, init='k-means++', n_init=1)
        self.kmeans.fit(self.standard_data)
        self.labels = self.kmeans.labels_
        self.cluster_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_) * self.direction_vector
        self._noisy_label = self._get_noisy_label()
        self.rejects = None
        self.thresholds = None

    @property
    def noisy_label(self):
        """Cluster label for noisy data"""
        return self._noisy_label

    @property
    def clean_labels(self):
        """Cluster labels for clean data"""
        return [idx for idx in range(self.k) if idx != self._noisy_label]

    @property
    def direction_vector(self):
        """Direction vector for the features (1 for CLEAN_LOW, -1 for CLEAN_HIGH)"""
        return np.array([1 if self.filters[name].score_direction == CLEAN_LOW else -1
                         for name in self.df.columns])

    def _get_noisy_label(self):
        """Find label for the noisy cluster"""
        means = np.mean(self.kmeans.cluster_centers_, axis=1)

        # Output some cluster information
        nlabels = Counter(self.labels)
        for i, (center, inv_center, mean) in enumerate(zip(self.kmeans.cluster_centers_, self.cluster_centers, means)):
            logger.info('Cluster #%s', i)
            logger.info('* number of samples: %s', nlabels[i])
            logger.info('* centroid (score, scaled value, original value):')
            for j, val in enumerate(center):
                logger.info('  %s\t%s\t%s', self.df.columns[j].ljust(25), round(val, 2), round(inv_center[j], 2))
            logger.info('Average center\t%s', np.round(mean, 2))

        # Cluster center of the noisiest cluster based on average features
        noisy_mean = np.max(means)
        noisy_label = np.argmax(means)
        logger.info('Cluster center of the noisiest cluster (%s)', np.round(noisy_mean, 2))
        logger.info('Noisy label: %s', noisy_label)
        noisy_labels = np.where(self.labels == noisy_label)[0]
        logger.info('Number of noisy labels: %s',
                    f'{len(noisy_labels)}/{len(self.labels)} ({round(100*len(noisy_labels)/len(self.labels), 2)}%)')
        return noisy_label

    def get_extreme_data_mean(self, data):
        """Return cleanest and noisiest parts based on mean"""
        l = len(data)
        low, high = int(0.1*l), int(0.9*l)
        data = np.array(sorted(data, key=lambda i: np.mean(i)))
        extreme_data = np.concatenate((data[:low], data[high:]))
        return extreme_data

    def get_extreme_data_euclidean(self, data):
        """Return extreme parts based on euclidean distance to extreme points"""
        extreme_points = []
        for i in range(len(data[0])):
            score_dim = data[...,i]
            extreme_points.append(data[np.argmax(score_dim)])
            extreme_points.append(data[np.argmin(score_dim)])
        n = int((len(data)*0.2)/len(extreme_points))
        mean = np.mean(data, 0)
        extreme_data = []
        for ep in extreme_points:
            dists = np.linalg.norm(data-ep, axis=1)
            i_mins = np.argpartition(dists, n)[:n]
            extreme_data.append(data[i_mins])
            data = np.delete(data, i_mins, axis=0)
        return np.vstack(extreme_data)

    def get_columns(self):
        """Return data column names"""
        return self.df.columns

    def get_thresholds(self, method='noisy_center', precision=6):
        """Return a list of thresholds for noisy samples"""
        if method != 'noisy_center':
            raise ValueError(f'Method {method} for thresholds not implemented')
        return self.cluster_centers[self.noisy_label].round(precision).tolist()

    def get_rejects(self):
        """Train random forest classifier to find important features

        Returns a list of booleans (True = reject).

        """
        logger.info('Training random forest')
        clf = RandomForestClassifier(random_state=1)
        clf.fit(self.standard_data, self.labels)
        logger.info('Finding important features')
        feature_importances = permutation_importance(clf, self.standard_data, self.labels)
        importance_mean_mean = np.mean(feature_importances.importances_mean)
        rej_coef = 0.1
        logger.info('* mean importance: %s', round(importance_mean_mean, 3))
        logger.info('* rejection coefficient: %s', rej_coef)
        logger.info('* decisions:')
        rejects = []
        for i, col in enumerate(self.df.columns):
            importance = feature_importances['importances_mean'][i]
            reject = importance < importance_mean_mean * rej_coef
            logger.info('  %s\t%s\t%s', col.ljust(25), round(importance, 3), 'reject' if reject else 'keep')
            rejects.append(reject)
        return rejects

    def get_result_df(self):
        """Return dataframe containing the thresholds and reject booleans"""
        self.rejects = self.get_rejects()
        self.thresholds = self.get_thresholds()
        return pd.DataFrame.from_dict(
            {'name': self.get_columns(), 'threshold': self.thresholds, 'reject': self.rejects})

    def plot(self, plt, apply_rejects=True, projection='pca'):
        """Plot clustering and histograms"""
        plt.figure(figsize=(10, 10))
        if projection == 'pca':
            projection = decomposition.PCA(n_components=2)
        elif projection == 'random':
            projection = random_projection.GaussianRandomProjection(n_components=2)
        else:
            raise ValueError(f"Unknown projection: {projection}")
        if apply_rejects and self.rejects:
            indices = [idx for idx, reject in enumerate(self.rejects) if not reject]
            data_t = projection.fit_transform(self.standard_data[:, indices])
            centroids = projection.transform(self.kmeans.cluster_centers_[:, indices])
        else:
            data_t = projection.fit_transform(self.standard_data)
            centroids = projection.transform(self.kmeans.cluster_centers_)
        for label_id in range(self.k):
            points = np.where(self.labels == label_id)
            plt.scatter(data_t[points, 0], data_t[points, 1],
                        c='orange' if label_id == self.noisy_label else 'blue',
                        label='noisy' if label_id == self.noisy_label else 'clean',
                        marker=',', s=1, alpha=0.1)
        for label_id in range(self.k):
            plt.scatter(centroids[label_id, 0], centroids[label_id, 1], s=100, alpha=1,
                        marker='+', c='brown' if label_id == self.noisy_label else 'darkblue',
                        label='noisy centroid' if label_id == self.noisy_label else 'clean centroid')
        plt.legend()
        plt.title('Clusters')
        noisy_samples = self.df.iloc[np.where(self.labels == self.noisy_label)]
        clean_samples = self.df.iloc[np.where(self.labels != self.noisy_label)]
        subplots_n = noisy_samples.hist(bins=100, figsize=(10, 10))
        plt.suptitle('Histograms for noisy samples')
        subplots_c = clean_samples.hist(bins=100, figsize=(10, 10))
        plt.suptitle('Histograms for clean samples')
        if apply_rejects and self.rejects:
            col_index = {col: idx for idx, col in enumerate(self.get_columns())}
            for axes in ([axes for sublist in subplots_c for axes in sublist] +
                         [axes for sublist in subplots_n for axes in sublist]):
                title = axes.get_title()
                if not title:
                    continue
                idx = col_index[title]
                if self.rejects[idx]:
                    axes.text(0.5, 1, 'REJECTED', horizontalalignment='center',
                              verticalalignment='top', transform=axes.transAxes, color='r')
                axes.axvline(self.thresholds[idx], color='k', linestyle='--', alpha=0.5)
