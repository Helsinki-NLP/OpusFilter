"""Unsupervised threshold selection for filters"""

from collections import Counter
import itertools
import logging
import os

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn import decomposition, preprocessing, random_projection
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted
import numpy as np

from . import CLEAN_LOW
from . import filters as filtermodule
from .classifier import load_dataframe


logger = logging.getLogger(__name__)


# allow scikit-learn style code here
# pylint: disable=C0103,W0613
class ArcProjection(BaseEstimator, TransformerMixin):
    """Project data to two dimensions using evenly distributed unit vectors

    The assigment of unit vectors to the original vectors is optimized
    so that L2 norm between the correlation coefficients of the
    original features and dot product of the vectors is minimized.

    Parameters
    ----------

    arc : float, 'auto' or 'full', default='auto'
        Length of the arc used for projection vectors. For 'auto', the
        length is set as the arccosine of the minimum correlation of
        the input features. For 'full', full circle (2 * pi) is used.

    Attributes
    ----------
    n_components_ : int
        Number of components, always 2.

    components_ : ndarray of shape (n_components, n_features)
        Matrix used for the projection.

    """

    def __init__(self, arc='auto'):
        self.components_ = None
        self.n_components_ = 2
        self.arc = arc

    def _best_permutation(self, current, n_features, eval_func):
        """Greedy search for best permutation of projection matrix"""
        permutation = np.arange(n_features)
        while True:
            logger.debug("current permutation: %s", permutation)
            minpair, mincost = self._best_single_swap(current, eval_func)
            logger.debug("best swap: %s (%s)", minpair, mincost)
            if minpair is None:
                return permutation, current, mincost
            fwd, rev = list(minpair), list(reversed(minpair))
            current[fwd] = current[rev]
            permutation[fwd] = permutation[rev]

    @staticmethod
    def _best_single_swap(matrix, eval_func):
        """Find best swap of matrix rows based in eval_func"""
        mincost = eval_func(matrix)
        minpair = None
        for pair in itertools.combinations(range(matrix.shape[0]), 2):
            fwd, rev = list(pair), list(reversed(pair))
            matrix[fwd] = matrix[rev]
            cost = eval_func(matrix)
            if cost < mincost:
                mincost = cost
                minpair = pair
            matrix[fwd] = matrix[rev]
        return minpair, mincost

    def _make_matrix(self, corrmat, n_features):
        """Generate the random projection matrix.

        Parameters
        ----------
        corrmat : ndarray of shape (n_features, n_features),
            Correlation matrix for the original features

        n_features : int,
            Dimensionality of the original source space.

        Returns
        -------
        components : ndarray of shape (n_components, n_features)
            The generated random matrix.
        """
        def costf(matrix):
            return ((corrmat - matrix @ matrix.T)**2).sum()

        if self.arc == 'full':
            dist = 2 * np.pi / n_features
        elif self.arc == 'auto':
            dist = np.arccos(corrmat.min()) / (n_features - 1)
        else:
            dist = self.arc / (n_features - 1)
        matrix = np.array([
            [np.cos(idx * dist), np.sin(idx * dist)] for idx in range(n_features)
        ])
        _, best_m, best_cost = self._best_permutation(matrix.copy(), n_features, costf)
        logger.debug("fcorr:\n%s", corrmat.round(3))
        logger.debug("vdist:\n%s", (best_m @ best_m.T).round(3))
        logger.debug("diff:\n%s", (corrmat - best_m @ best_m.T).round(3))
        logger.debug("cost: %s", best_cost)
        return best_m.T

    def fit(self, X: np.array, y=None):
        """Generate a projection matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training set: only the shape is used to find matrix dimensions.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            BaseRandomProjection class instance.
        """
        X = self._validate_data(
            X, accept_sparse=["csr", "csc"], dtype=[np.float64, np.float32]
        )

        n_features = X.shape[1]
        corrmat = np.corrcoef(X.T)
        self.components_ = self._make_matrix(
            corrmat, n_features
        ).astype(X.dtype, copy=False)

        return self

    def transform(self, X):
        """Project the data by using matrix product with the random matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data to project into a smaller dimensional space.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projected array.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X, accept_sparse=["csr", "csc"], reset=False, dtype=[np.float64, np.float32]
        )

        return X @ self.components_.T
# pylint: enable=C0103,W0613


# Would need some refactoring, but disable pylint's "too many" warnings for now.
# pylint: disable=R0902,R0912,R0914,R0915
class ScoreClusters:
    """Cluster segments by filter scores

    Train k-means clustering and take thresholds based on the noisy cluster center.

    """

    def __init__(self, score_file, k=2):
        self.k = k
        self.df = load_dataframe(score_file)
        self.filters = {}
        for name in self.df.columns:
            first_part = name.split('.')[0]
            filter_cls = getattr(filtermodule, first_part)
            self.filters[name] = filter_cls
        self.scaler = preprocessing.StandardScaler()
        self.standard_data = self.scaler.fit_transform(self.df.mul(self.direction_vector))

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

    def plot(self, plt, path=None, apply_rejects=True, projection='arc'):
        """Plot clustering and histograms"""
        if projection == 'pca':
            proj = decomposition.PCA(n_components=2)
        elif projection == 'random':
            proj = random_projection.GaussianRandomProjection(n_components=2)
        elif projection == 'arc':
            proj = ArcProjection()
        else:
            raise ValueError(f"Unknown projection: {projection}")
        col_index = {col: idx for idx, col in enumerate(self.get_columns())}
        index_col = dict(enumerate(self.get_columns()))
        if apply_rejects and self.rejects:
            indices = [idx for idx, reject in enumerate(self.rejects) if not reject]
            cols = [index_col[idx] for idx in indices]
            index_col = dict(enumerate(cols))
            data_t = proj.fit_transform(self.standard_data[:, indices])
            centroids = proj.transform(self.kmeans.cluster_centers_[:, indices])
        else:
            data_t = proj.fit_transform(self.standard_data)
            centroids = proj.transform(self.kmeans.cluster_centers_)
        plt.figure()
        for idx in range(proj.components_.shape[1]):
            plt.arrow(0, 0, proj.components_[0, idx], proj.components_[1, idx], head_width=0.01, fc='k')
            plt.text(proj.components_[0, idx], proj.components_[1, idx], index_col[idx])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Projection vectors for clustering')
        if path is not None:
            plt.savefig(os.path.join(path, 'projection.pdf'))
        plt.figure(figsize=(10, 10))
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
        plt.title('Clustering')
        if path is not None:
            plt.savefig(os.path.join(path, 'clustering.pdf'))
        noisy_samples = self.df.iloc[np.where(self.labels == self.noisy_label)]
        clean_samples = self.df.iloc[np.where(self.labels != self.noisy_label)]
        subplots_n = noisy_samples.hist(bins=100, figsize=(10, 10))
        fig_noisy = plt.gcf()
        plt.suptitle('Histograms for noisy samples')
        subplots_c = clean_samples.hist(bins=100, figsize=(10, 10))
        fig_clean = plt.gcf()
        plt.suptitle('Histograms for clean samples')
        if apply_rejects and self.rejects:
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
        if path is not None:
            fig_clean.savefig(os.path.join(path, 'histogram_clean.pdf'))
            fig_noisy.savefig(os.path.join(path, 'histogram_noisy.pdf'))
# pylint: enable=R0902,R0912,R0914,R0915
