"""Unsupervised threshold selection for filters"""

from collections import Counter
import logging

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
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

    def __init__(self, score_file, n=2):
        self.df = load_dataframe(score_file)
        self.filters = {}
        for name in self.df.columns:
            first_part = name.split('.')[0]
            filter_cls = getattr(filtermodule, first_part)
            self.filters[name] = filter_cls
        self.scaler = preprocessing.StandardScaler()
        self.standard_data = self.scaler.fit_transform(self.df)

        logger.info('Training KMeans with %s clusters', n)
        self.kmeans = KMeans(n_clusters=n, random_state=0, n_init='auto').fit(self.standard_data)
        self.labels = self.kmeans.labels_
        self.cluster_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        self.noisy_label = self._get_noisy_label()
        self.clean_label = np.abs(self.noisy_label - 1)

    def _get_noisy_label(self):
        """Find label for the noisy cluster"""
        centers = self.kmeans.cluster_centers_
        inv_centers = self.cluster_centers

        # Flip values if low score indicates clean data
        dir_fixed_centers = []
        for center in centers:
            fixed_center = []
            for j, name in enumerate(self.df.columns):
                value = center[j].copy()
                if self.filters[name].score_direction == CLEAN_LOW:
                    value *= -1
                fixed_center.append(value)
            dir_fixed_centers.append(fixed_center)
        means = np.mean(dir_fixed_centers, axis=1)

        nlabels = Counter(self.labels)
        for k, (center, inv_center, mean) in enumerate(zip(centers, inv_centers, means)):
            logger.info('Cluster #%s - number of samples: %s', k, nlabels[k])
            for j, val in enumerate(center):
                logger.info('%s\t%s\t%s', self.df.columns[j], round(val, 2), round(inv_center[j], 2))
            logger.info('Average center\t%s', np.round(mean, 2))

        # Cluster center of the noisiest cluster based on average features
        noisy_mean = np.min(means)
        noisy_label = np.argmin(means)
        logger.info('Cluster center of the noisiest cluster (%s)', np.round(noisy_mean, 2))
        logger.info('Noisy label: %s', noisy_label)
        noisy_labels = np.where(self.labels == noisy_label)[0]
        logger.info('Number of noisy labels: %s',
                    f'{len(noisy_labels)}/{len(self.labels)} ({round(100*len(noisy_labels)/len(self.labels), 2)}%)')
        return noisy_label

    def get_thresholds(self, method='noisy_center', precision=6):
        """Return thresholds for noisy samples"""
        if method != 'noisy_center':
            raise ValueError(f'Method {method} for thresholds not implemented')
        return self.cluster_centers[self.noisy_label].round(precision).tolist()

    def get_rejects(self):
        """Train random forest classifier to find important features"""
        logger.info('Training random forest')
        clf = RandomForestClassifier(random_state=1)
        clf.fit(self.standard_data, self.labels)
        logger.info('Finding important features')
        feature_importances = permutation_importance(clf, self.standard_data, self.labels)
        importance_mean_mean = np.mean(feature_importances.importances_mean)
        rej_coef = 0.1
        logger.info('mean importance: %s', round(importance_mean_mean, 3))
        logger.info('rejection coefficient: %s', rej_coef)
        rejects = {}
        for i, k in enumerate(self.df.columns):
            importance = feature_importances['importances_mean'][i]
            rejects[k] = importance < importance_mean_mean * rej_coef
            logger.info('%s\t%s\t%s', k, round(importance, 3), 'reject' if rejects[k] else 'keep')
        return rejects

    def plot(self, plt):
        """Plot clustering and histograms"""
        plt.figure(figsize=(10, 10))
        data_t = PCA(n_components=2).fit_transform(self.standard_data)
        colors = ['orange' if lbl == self.noisy_label else 'blue' for lbl in self.labels]
        plt.scatter(data_t[:, 0], data_t[:, 1], c=colors, marker=',', s=1)
        plt.title('Clusters')
        noisy_samples = self.df.iloc[np.where(self.labels == self.noisy_label)]
        clean_samples = self.df.iloc[np.where(self.labels == self.clean_label)]
        noisy_samples.hist(bins=100, figsize=(10, 10))
        plt.suptitle('Histograms for noisy samples')
        clean_samples.hist(bins=100, figsize=(10, 10))
        plt.suptitle('Histograms for clean samples')
