"""Unsupervised threshold selection for filters"""

import inspect
import os
import shutil
from collections import Counter
import logging
import pathlib
import tempfile

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
import ruamel.yaml

from . import CLEAN_LOW
from . import filters as filtermodule
from .autogen import ConfigurationGenerator
from .classifier import load_dataframe
from .opusfilter import OpusFilter


logger = logging.getLogger(__name__)
yaml = ruamel.yaml.YAML()


class ScoreClusters:
    """Cluster segments by filter scores"""

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

        noisy_label, thresholds = self._get_noisy_label_and_thresholds()
        self.noisy_label = noisy_label
        self.clean_label = np.abs(self.noisy_label - 1)
        self.thresholds = thresholds

    def _get_noisy_label_and_thresholds(self):
        """Find filter thresholds

        Train k-means clustering and take thresholds from the noisy cluster center.

        """
        centers = self.kmeans.cluster_centers_
        inv_centers = self.scaler.inverse_transform(centers)

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
        thresholds = inv_centers[noisy_label].round(3).tolist()
        return noisy_label, thresholds

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


class FilterThresholdFinder:
    """Find thresholds for filters based on score clustering"""

    def __init__(self, files, langs, scripts, sample_size, inter_dir, overwrite):
        self.input_files = files
        self.sample_size = sample_size
        self.langs = langs
        self.scripts = scripts
        if inter_dir:
            self.use_tmp = False
            self.inter_dir = inter_dir
        else:
            self.use_tmp = True
            self.inter_dir = tempfile.mkdtemp()
        self.label_file_path = os.path.join(self.inter_dir, 'labels.txt')
        self.overwrite = overwrite
        self.filter_params = {
            'AlphabetRatioFilter': {},
            'LengthRatioFilter.char': {
                'name': 'char',
                'unit': 'char'},
            'LengthRatioFilter.word': {
                'name': 'word',
                'unit': 'word'},
            'NonZeroNumeralsFilter': {},
        }
        if self.langs:
            self.filter_params['LanguageIDFilter'] = {
                'name': 'cld2',
                'id_method': 'cld2',
                'languages': langs
            }
        if self.scripts:
            self.filter_params['CharacterScoreFilter'] = {'scripts': self.scripts}
        if len(self.input_files) == 2:
            self.filter_params['TerminalPunctuationFilter'] = {}

    def find_thresholds(self):
        """Find suitable filter thresholds

        Returns a dict of filter parameters and a ScoreClusters object

        """
        score_file = self._prepare_data()
        scoreclusters = ScoreClusters(os.path.join(self.inter_dir, score_file))
        self._set_parameters(scoreclusters.thresholds, scoreclusters.get_rejects())
        if os.path.isfile(self.label_file_path) and not self.overwrite:
            logger.info('Label file "%s" exits, not overwriting', self.label_file_path)
        else:
            with open(self.label_file_path, 'w', encoding='utf-8') as label_file:
                for label in scoreclusters.labels:
                    label_file.write(str(label)+'\n')
        if self.use_tmp:
            shutil.rmtree(self.inter_dir)
        filters = [{k.split('.', maxsplit=1)[0]: v} for k, v in self.filter_params.items()]
        return filters, scoreclusters

    def _prepare_data(self):
        """Remove duplicates and empty lines, take a sample of size n, produce filter scores"""
        config_gen = ConfigurationGenerator(files=[os.path.abspath(f) for f in self.input_files], workdir=self.inter_dir)
        config_gen.add_remove_duplicates()
        config_gen.add_filter([{'LengthFilter': {'unit': 'word', 'min_length': 1, 'max_length': 150}}])
        config_gen.add_subset(self.sample_size, 1)
        score_file = config_gen.add_score([{k.split('.', maxsplit=1)[0]: v} for k, v in self.filter_params.items()])
        pre_config = config_gen.get_config()
        yaml.dump(pre_config, pathlib.Path(os.path.join(self.inter_dir, 'config.yaml')))
        opusf = OpusFilter(pre_config)
        opusf.execute_steps(overwrite=self.overwrite)
        return score_file

    def _set_parameters(self, thresholds, rejects):
        """Set filter parameters based on thresholds and rejects"""
        for i, name in enumerate(rejects):
            fullname = name
            name_parts = name.split('.')
            filter_name = name_parts[0]
            filter_cls = getattr(filtermodule, filter_name)
            filt_args = inspect.signature(filter_cls).parameters
            endp = name_parts[-1]
            if endp.isnumeric():
                # numeric last part is language index
                name = '.'.join(name_parts[:-1])
            if 'thresholds' in filt_args:
                parameter = self.filter_params.get(filter_name)
                if 'thresholds' not in parameter:
                    parameter['thresholds'] = []
                if rejects[fullname]:
                    # FIXME: -1 may not work for all filters
                    parameter['thresholds'].insert(int(endp), -1)
                else:
                    parameter['thresholds'].insert(int(endp), thresholds[i])
                if len(parameter['thresholds']) == 2:
                    if all(v == -1 for v in parameter['thresholds']):
                        del self.filter_params[filter_name]
            elif 'threshold' in filt_args:
                parameter = self.filter_params.get(name)
                if rejects[fullname]:
                    if name in self.filter_params:
                        del self.filter_params[name]
                    continue
                if parameter is None:
                    continue
                prev_t = parameter.get('threshold')
                if prev_t is None or thresholds[i] < prev_t:
                    parameter['threshold'] = thresholds[i]
