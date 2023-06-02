import os
from collections import Counter
import logging
import pprint

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import ruamel.yaml
import pandas as pd

from opusfilter.classifier import load_dataframe
from opusfilter.filters import AlphabetRatioFilter, CharacterScoreFilter, LanguageIDFilter, LengthRatioFilter, NonZeroNumeralsFilter, TerminalPunctuationFilter
from opusfilter.opusfilter import OpusFilter
from . import CLEAN_LOW

plt.style.use('seaborn')

logger = logging.getLogger(__name__)

class ConfigGenerator:

    def __init__(self, files, langs, scripts, output_file, sample_size, work_dir, graph, overwrite):
        self.input_files = files
        self.sample_size = sample_size

        self.base_name = files[0]
        for l in langs:
            self.base_name = self.base_name.replace(l, '')

        self.config_input_files = ['noemp_'+f for f in self.input_files]
        self.langs = langs
        self.work_dir = work_dir
        self.output_config = output_file
        self.graph = graph
        self.overwrite = overwrite

        self.filter_params = {
                'AlphabetRatioFilter': {},
                'LanguageIDFilter': {
                    'id_method': 'cld2',
                    'languages': langs},
                'LengthRatioFilter.char': {
                    'name': 'char',
                    'unit': 'char'},
                'LengthRatioFilter.word': {
                    'name': 'word',
                    'unit': 'word'},
                'NonZeroNumeralsFilter': {},
                'TerminalPunctuationFilter': {}
                }

        if scripts:
            self.filter_params['CharacterScoreFilter'] = {'scripts': scripts}

    def generate_config(self):
        if os.path.isfile(self.output_config) and not self.overwrite:
            logger.info(f'Output file "{self.output_config}" exists, not overwriting')
            exit()

        score_file, sample_files = self.prepare_data(self.sample_size)

        df = load_dataframe(os.path.join(self.work_dir, score_file))
        thresholds, standard_data, labels, label_file_name = self.find_thresholds(df, 2)

        rejects = self.get_rejects(standard_data, labels, df.columns)

        self.make_config_yaml(thresholds, rejects)

        self.sort(label_file_name, score_file, sample_files)

        if self.graph:
            plt.show()

    def prepare_data(self, n):
        # Remove duplicates and empty lines, take a sample of size n, produce filter scores

        dedup_files = ['dedup_'+f for f in self.input_files]
        sample_files = ['sample_'+f for f in self.input_files]
        score_file = f'scores_{self.base_name}.{"-".join(self.langs)}.jsonl.gz'
        score_file = score_file.replace('..', '.')

        pre_config =  {'common': {'output_directory': self.work_dir},
                    'steps': [
                        {'type': 'remove_duplicates',
                        'parameters': {
                            'inputs': self.input_files,
                            'outputs': dedup_files}
                        },
                        {'type': 'filter',
                        'parameters': {
                            'inputs': dedup_files,
                            'outputs': self.config_input_files,
                            'filters': [
                                {'LengthFilter':
                                    {'unit': 'word', 'min_length': 1, 'max_length': 150}
                                }]
                            }
                        },
                        {'type': 'subset',
                        'parameters': {
                            'inputs': self.config_input_files,
                            'outputs': sample_files,
                            'size': n,
                            'seed': 1}
                        },
                        {'type': 'score',
                        'parameters': {
                            'inputs': sample_files,
                            'output': score_file,
                            'filters': [{k.split('.')[0]: v} for k, v in self.filter_params.items()]
                            }
                        }
                    ]
                }

        of = OpusFilter(pre_config)
        of.execute_steps(overwrite=self.overwrite)

        return score_file, sample_files

    def find_thresholds(self, df, n):
        # Find filter thresholds: train Kmeans clustering with n cluster
        # and take parameters from the noisy cluster center

        filters = [AlphabetRatioFilter, CharacterScoreFilter, LanguageIDFilter,
                LengthRatioFilter, NonZeroNumeralsFilter, TerminalPunctuationFilter]
        filters = {f.__name__: f for f in filters}

        # Remove unused filters
        for name in df.columns:
            first_part = name.split('.')[0]
            if first_part not in filters.keys():
                del df[name]

        scaler = preprocessing.StandardScaler()
        standard_data = scaler.fit_transform(df)

        logger.info(f'Training KMeans with {n} clusters')
        kmeans = KMeans(n_clusters=n, random_state=0).fit(standard_data)

        centers = kmeans.cluster_centers_

        # Flip values if low score indicates clean data
        dir_fixed_centers = []
        for center in centers:
            fixed_center = []
            for j, name in enumerate(df.columns):
                first_part = name.split('.')[0]
                value = center[j].copy()
                if filters[first_part].score_direction == CLEAN_LOW:
                    value *= -1
                fixed_center.append(value)
            dir_fixed_centers.append(fixed_center)

        means = np.round(np.mean(dir_fixed_centers, axis=1), 2)
        i_centers = scaler.inverse_transform(centers)
        nlabels = Counter(kmeans.labels_)

        k = 0
        for c, i_c, m in zip(centers, i_centers, means):
            print(f'Number of samples: {nlabels[k]}')
            k += 1
            for j, v in enumerate(c):
                print(df.columns[j][:10], round(v, 2), round(i_c[j], 2), sep='\t')
            print(f'Average center\t{m}\n')

        low_mean = np.min(means)

        # Cluster center of the noisiest cluster based on average features
        noisy_label = np.argmin(means)
        clean_label = np.abs(noisy_label-1)
        thresholds = i_centers[noisy_label].round(3).tolist()
        labels = kmeans.labels_

        if self.graph:
            fig = plt.figure(figsize=(10,10))
            X_t = self.pca_data(standard_data)
            colors = ['orange' if l == noisy_label else 'blue' for l in labels]
            plt.scatter(X_t[:,0], X_t[:,1], c=colors, marker=',', s=1)

        label_file_name = f'{self.base_name}.labels.txt'
        label_file_name = label_file_name.replace('..', '.')
        label_file_path = os.path.join(self.work_dir, label_file_name)

        if os.path.isfile(label_file_path) and not self.overwrite:
            logger.info(f'Label file "{label_file_name}" exits, not overwriting')
        else:
            with open(label_file_path, 'w') as label_file:
                for label in labels:
                    label_file.write(str(label)+'\n')

        logger.info(f'Cluster center of the noisiest cluster ({low_mean})')
        logger.info(f'Noisy label: {noisy_label}')
        noisy_labels = np.where(labels == noisy_label)[0]
        logger.info(f'N noisy labels: {len(noisy_labels)}/{len(labels)} ({round(100*len(noisy_labels)/len(labels), 2)}%)')

        if self.graph:
            noisy_samples = df.iloc[np.where(labels==noisy_label)]
            clean_samples = df.iloc[np.where(labels==clean_label)]
            noisy_samples.hist(bins=100, figsize=(10,10))
            clean_samples.hist(bins=100, figsize=(10,10))

        return thresholds, standard_data, labels, label_file_name

    def get_rejects(self, X, labels, columns):
        # Train random forest and find important features

        logger.info('Training random forest')
        clf = RandomForestClassifier(random_state=1)
        clf.fit(X, labels)

        logger.info('Finding important features')
        feature_importances = permutation_importance(clf, X, labels)
        importance_mean_mean = np.mean(feature_importances.importances_mean)
        rej_coef = 0.1

        print(f'mean importance: {round(importance_mean_mean, 3)}')
        print(f'rejection coefficient: {rej_coef}\n')
        rejects = {}
        for i, k in enumerate(columns):
            importance = feature_importances['importances_mean'][i]
            rejects[k] = importance < importance_mean_mean * rej_coef
            print(k[:10], round(importance, 3), 'reject' if rejects[k] else 'keep', sep='\t')

        return rejects

    def make_config_yaml(self, thresholds, rejects):
        for i, name in enumerate(rejects.keys()):
            fullname = name
            name_parts = name.split('.')
            stap = name_parts[0]
            endp = name_parts[-1]
            filt_args = eval(stap+'.__init__.__code__.co_varnames')
            if endp.isnumeric():
                name = '.'.join(name_parts[:-1])
            if 'thresholds' in filt_args:
                parameter = self.filter_params.get(stap)
                if 'thresholds' not in parameter.keys():
                    parameter['thresholds'] = []
                if rejects[fullname]:
                    parameter['thresholds'].insert(int(endp), 0)
                else:
                    parameter['thresholds'].insert(int(endp), thresholds[i])
                if len(parameter['thresholds']) == 2:
                    if all(v == 0 for v in parameter['thresholds']):
                        del self.filter_params[stap]
            elif 'threshold' in filt_args:
                parameter = self.filter_params.get(name)
                if rejects[fullname]:
                    if name in self.filter_params.keys():
                        del self.filter_params[name]
                    continue
                if parameter == None:
                    continue
                prev_t = parameter.get('threshold')
                if prev_t == None or thresholds[i] < prev_t:
                    parameter['threshold'] = thresholds[i]

        output_files = ['filtered_'+f for f in self.input_files]
        out_config = {'common':
                        {'output_directory': self.work_dir},
                    'steps':
                        [{'type': 'filter',
                        'parameters':
                            {'inputs': self.input_files,
                            'outputs': output_files,
                            'filters': [{k.split('.')[0]: v} for k, v in self.filter_params.items()]
                            }
                        }]
                    }

        logger.info('Generated config file:')
        pprint.pprint(out_config)

        yaml = ruamel.yaml.YAML()
        with open(self.output_config, 'w') as out_conf:
            yaml.dump(out_config, out_conf)

    def sort(self, labels, score_file, sample_files):
        input_files = sample_files + [labels, score_file]
        output_files = ['sorted_'+n for n in input_files]
        sort_config = {'common': {'output_directory': self.work_dir},
                'steps': [
                    {'type': 'sort',
                    'parameters': {
                        'inputs': input_files,
                        'outputs': output_files,
                        'values': labels}}
                    ]
                }
        of = OpusFilter(sort_config)
        of.execute_steps(overwrite=self.overwrite)

    def pca_data(self, X):
        # PCA to get 2d graph for clusters
        pca = PCA(n_components=2)
        pca.fit(X)
        X_t = pca.transform(X)
        return X_t

