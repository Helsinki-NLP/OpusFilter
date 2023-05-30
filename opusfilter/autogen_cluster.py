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

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

class ConfigGenerator:

    def __init__(self, files, langs, scripts, output_dir, output_file, graph=True):
        self.input_files = files
        self.langs = langs
        self.output_dir = output_dir
        self.output_config = output_file
        self.graph = graph

        self.filter_params = {
                'AlphabetRatioFilter': {},
                'CharacterScoreFilter': {
                    'scripts': scripts},
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

    def prepare_data(self):
        # Remove duplicates and empty lines, take a 100k samples, produce filter scores
        dedup_files = ['dedup_'+f for f in self.input_files]
        noemp_files = ['noemp_'+f for f in self.input_files]
        sample_files = ['100k_'+f for f in self.input_files]
        score_name = sample_files[0]
        for l in self.langs:
            score_name = score_name.replace(l, '')
        score_file = f'scores_{score_name}.{"-".join(self.langs)}.jsonl.gz'
        score_file = score_file.replace('..', '.')

        pre_config =  {'common': {'output_directory': self.output_dir},
                    'steps': [
                        {'type': 'remove_duplicates',
                        'parameters': {
                            'inputs': self.input_files,
                            'outputs': dedup_files}
                        },
                        {'type': 'filter',
                        'parameters': {
                            'inputs': dedup_files,
                            'outputs': noemp_files,
                            'filters': [
                                {'LengthFilter':
                                    {'unit': 'word', 'min_length': 1, 'max_length': 150}
                                }]
                            }
                        },
                        {'type': 'subset',
                        'parameters': {
                            'inputs': noemp_files,
                            'outputs': sample_files,
                            'size': 100000,
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
        of.execute_steps(overwrite=False)

        return score_file, noemp_files, sample_files

    def sort(self, labels, score_file, sample_files):
        input_files = sample_files + [labels, score_file]
        output_files = ['sorted_'+n for n in input_files]
        sort_config = {'common': {'output_directory': self.output_dir},
                'steps': [
                    {'type': 'sort',
                    'parameters': {
                        'inputs': input_files,
                        'outputs': output_files,
                        'values': labels}}
                    ]
                }
        of = OpusFilter(sort_config)
        of.execute_steps(overwrite=False)

    def generate_config(self):
        score_file, noemp_files, sample_files = self.prepare_data()

        df = load_dataframe(os.path.join(self.output_dir, score_file))

        filters = [AlphabetRatioFilter, CharacterScoreFilter, LanguageIDFilter, LengthRatioFilter, NonZeroNumeralsFilter, TerminalPunctuationFilter]
        filters = {f.__name__: f for f in filters}

        for name in df.columns:
            first_part = name.split('.')[0]
            if first_part not in filters.keys():
                del df[name]

        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(df)

        n_clusters_range = (2, 3)

        if self.graph:
            X_t, pca, low_b, nrows, ncols = self.prepare_graph(X, n_clusters_range)

        # Find thresholds
        thresholds = None
        lowest_mean = None
        labels = None
        noisy_label = None
        for i in range(*n_clusters_range):
            logger.info(f'Training KMeans with {i} clusters')
            kmeans = KMeans(n_clusters=i, random_state=0).fit(X)

            centers = kmeans.cluster_centers_

            dir_fixed_centers = []
            for center in centers:
                fixed_center = []
                for j, name in enumerate(df.columns):
                    first_part = name.split('.')[0]

                    value = center[j].copy()
                    if filters[first_part].score_direction == 'clean_low':
                        value *= -1
                    fixed_center.append(value)
                dir_fixed_centers.append(fixed_center)

            means = np.round(np.mean(dir_fixed_centers, axis=1), 2)

            i_centers = scaler.inverse_transform(centers)

            pca_centers = pca.transform(centers)
            noisy_pca_center = pca_centers[np.argmin(means)]
            nlabels = Counter(kmeans.labels_)

            k = 0
            for c, i_c, p_c, m in zip(centers, i_centers, pca_centers, means):
                print(f'Center: {p_c}')
                print(f'Number of samples: {nlabels[k]}')
                k += 1
                for j, v in enumerate(c):
                    print(df.columns[j][:10], round(v, 2), round(i_c[j], 2), sep='\t')
                print(f'Average center\t{m}\n')

            low_mean = np.min(means)
            if not thresholds or low_mean < lowest_mean:
                # Cluster center of the noisiest cluster based on average features
                noisy_label = np.argmin(means)
                clean_label = np.abs(noisy_label-1)
                thresholds = i_centers[noisy_label].round(3).tolist()
                lowest_mean = low_mean
                labels = kmeans.labels_

            if self.graph:
                self.add_scatter(X_t, nrows, ncols, i-low_b+1, kmeans.labels_, pca_centers, noisy_pca_center, noisy_label)

        label_file_name = 'labels.txt'
        with open(os.path.join(self.output_dir, label_file_name), 'w') as label_file:
            for label in labels:
                label_file.write(str(label)+'\n')

        logger.info(f'Cluster center of the noisiest cluster ({lowest_mean})')
        logger.info(f'Noisy label: {noisy_label}')
        noisy_labels = np.where(labels == noisy_label)[0]
        logger.info(f'N noisy labels: {len(noisy_labels)}/{len(labels)} ({round(100*len(noisy_labels)/len(labels), 2)}%)')


        #ds = euclidean_distances(centers, df)
        noisy_samples = df.iloc[np.where(labels==noisy_label)]
        clean_samples = df.iloc[np.where(labels==clean_label)]
        if self.graph:
            noisy_samples.hist(bins=100, figsize=(10,10))
            clean_samples.hist(bins=100, figsize=(10,10))

        logger.info('Training random forest')
        clf = RandomForestClassifier(random_state=1)
        clf.fit(X, labels)
        #feature_importances = clf.feature_importances_[i]
        logger.info('Finding important features')
        feature_importances = permutation_importance(clf, X, labels)
        importance_mean_mean = np.mean(feature_importances.importances_mean)
        rej_coef = 0.1

        print(f'mean importance: {round(importance_mean_mean, 3)}')
        print(f'rejection coefficient: {rej_coef}\n')
        rejects = {}
        for i, k in enumerate(df.keys()):
            importance = feature_importances['importances_mean'][i]
            rejects[k] = importance < importance_mean_mean * rej_coef
            print(k[:10], round(importance, 3), 'reject' if rejects[k] else 'keep', sep='\t')

        #print(f'std importance: {np.std(feature_importances.importances_mean)}')

        # Generate config
        for i, name in enumerate(df.columns):
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
                    del self.filter_params[name]
                    continue
                if parameter == None:
                    continue
                prev_t = parameter.get('threshold')
                if prev_t == None or thresholds[i] < prev_t:
                    parameter['threshold'] = thresholds[i]

        output_files = ['filtered_'+f for f in self.input_files]
        out_config = {'common':
                        {'output_directory': self.output_dir},
                    'steps':
                        [{'type': 'filter',
                        'parameters':
                            {'inputs': noemp_files,
                            'outputs': output_files,
                            'filters': [{k.split('.')[0]: v} for k, v in self.filter_params.items()]
                            }
                        }]
                    }

        logger.info('Generated config file:')
        pprint.pprint(out_config)

        yaml = ruamel.yaml.YAML()
        yaml.dump(out_config, self.output_config)

        self.sort(label_file_name, score_file, sample_files)

        if self.graph:
            plt.show()

    def prepare_graph(self, X, n_clusters_range):
        pca = PCA(n_components=2)
        pca.fit(X)
        X_t = pca.transform(X)

        low_b = n_clusters_range[0]
        n_trains = n_clusters_range[1]-n_clusters_range[0]
        nrows, ncols = 1, 1
        prev = 'y'
        for i in range(1, n_trains+1):
            if i > nrows * ncols:
                if prev == 'y':
                    ncols += 1
                    prev = 'x'
                elif prev == 'x':
                    nrows += 1
                    prev = 'y'

        fig = plt.figure(figsize=(10,10))

        plt.style.use('seaborn')

        return X_t, pca, low_b, nrows, ncols

    def add_scatter(self, X_t, nrows, ncols, position, labels, pca_centers, noisy_pca_center, noisy_label):
        colors = ['orange' if l == noisy_label else 'blue' for l in labels]
        a = plt.subplot(nrows, ncols, position)
        a.scatter(X_t[:,0], X_t[:,1], c=colors, edgecolors='black') #marker=',', s=1)
        #a.scatter(pca_centers[:,0], pca_centers[:,1], color='blue', edgecolors='black')
        #a.scatter(noisy_pca_center[0], noisy_pca_center[1], color='red', edgecolors='black')

