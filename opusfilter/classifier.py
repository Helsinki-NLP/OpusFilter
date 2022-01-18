"""Filter classifier"""

import json
import logging
import collections
import functools
import math
import scipy.optimize

import numpy as np
import pandas as pd
from pandas import json_normalize
import sklearn.linear_model
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, log_loss

from . import grouper
from .util import file_open

logger = logging.getLogger(__name__)


def lists_to_dicts(obj):
    """Convert lists in a JSON-style object to dicts recursively

    Examples:

    >>> lists_to_dicts([3, 4])
    {"0": 3, "1": 4}
    >>> lists_to_dicts([3, [4, 5]])
    {"0": 3, "1": {"0": 4, "1": 5}}
    >>> lists_to_dicts({"a": [3, 4], "b": []})
    {"a": {"0": 3, "1": 4}, "b": {}}

    """
    if isinstance(obj, dict):
        return {key: lists_to_dicts(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return {str(idx): lists_to_dicts(value) for idx, value in enumerate(obj)}
    return obj


def load_dataframe(data_file):
    """Load normalized scores dataframe from a JSON lines file"""
    data = []
    with file_open(data_file) as dfile:
        for line in dfile:
            try:
                data.append(lists_to_dicts(json.loads(line)))
            except json.decoder.JSONDecodeError as err:
                logger.error(line)
                raise err
    return pd.DataFrame(json_normalize(data))


def load_dataframe_in_chunks(data_file, chunksize):
    """Yield normalized scores dataframes from a chunked JSON lines file

    Use instead of load_dataframe if the data is too large to fit in memory.

    """
    with file_open(data_file) as dfile:
        for num, chunk in enumerate(grouper(dfile, chunksize)):
            data = []
            for line in chunk:
                try:
                    data.append(lists_to_dicts(json.loads(line)))
                except json.decoder.JSONDecodeError as err:
                    logger.error(line)
                    raise err
            logger.info("Processing chunk %s with %s lines", num, len(data))
            yield pd.DataFrame(json_normalize(data))


def standardize_dataframe_scores(dataframe, features, means_stds=None):
    """Normalize, zero average, and set direction for scores in each column"""
    new_df = pd.DataFrame()
    if not means_stds:
        means_stds = {}
        for column in dataframe:
            vect = dataframe[column].to_numpy()
            if features[column].get('clean-direction', 'high') == 'low':
                direction = -1
            else:
                direction = 1
            means_stds[column] = (vect.mean(), vect.std(), direction)
    for column in features:
        vect = dataframe[column].to_numpy()
        mean, std, direction = means_stds[column]
        if std == 0:
            vect = [0 for i in range(len(dataframe[column]))]
        else:
            vect = direction * (vect - mean) / std
        new_df[column] = vect
    return new_df, means_stds


class Classifier:
    """Wrapper for sklearn classifiers (e.g. LogisticRegression)

    Includes feature selection and standardization from pandas
    dataframes.

    """

    def __init__(self, classname, params, features, standardize_params):
        self.classname = classname
        cls = getattr(sklearn.linear_model, self.classname)
        self.classifier = cls(**params)
        self.features = features
        self.standardize_params = standardize_params

    def standardize(self, dataframe):
        """Standardize features in the data frame"""
        if not self.standardize_params:
            logger.warning("Feature standardization parameters missing")
            return dataframe[self.features]
        return standardize_dataframe_scores(dataframe, self.features, self.standardize_params)[0]

    def train(self, dataframe, labels, standardize=True):
        """Train logistic regression with training_data"""
        dataframe = self.standardize(dataframe) if standardize else dataframe
        self.classifier.fit(dataframe[self.features], labels)

    def write_preds(self, input_fname, output_fname, true_label=None,
                    standardize=True, chunksize=None):
        """Write predicted class labels to output file"""
        if chunksize:
            dfs_tbc = load_dataframe_in_chunks(input_fname, chunksize)
        else:
            dfs_tbc = [load_dataframe(input_fname)]
        logger.info("Classifier labels: %s", self.classifier.classes_)
        with file_open(output_fname, 'w') as output:
            for df_tbc in dfs_tbc:
                df_std = self.standardize(df_tbc) if standardize else df_tbc
                labels = self.classifier.predict(df_std[self.features])
                if true_label:
                    true_labels = df_tbc[true_label]
                    logger.info('accuracy: %s', accuracy_score(true_labels, labels))
                    logger.info('confusion matrix:\n%s', confusion_matrix(true_labels, labels))
                for label in labels:
                    output.write(f'{label}\n')

    def write_probs(self, input_fname, output_fname, true_label=None,
                    standardize=True, chunksize=None):
        """Write classification probabilities to output file"""
        if chunksize:
            dfs_tbc = load_dataframe_in_chunks(input_fname, chunksize)
        else:
            dfs_tbc = [load_dataframe(input_fname)]
        logger.info("Classifier labels: %s", self.classifier.classes_)
        with file_open(output_fname, 'w') as output:
            for df_tbc in dfs_tbc:
                df_std = self.standardize(df_tbc) if standardize else df_tbc
                probas = self.classifier.predict_proba(df_std[self.features])
                if true_label:
                    true_labels = df_tbc[true_label]
                    logger.info('roc_auc: %s', roc_auc_score(true_labels, probas[:, 1]))
                for proba in probas[:, 1]:
                    output.write(f'{proba:.10f}\n')

    def weights(self):
        """Yield classifier weights"""
        if self.classname == "LogisticRegression":
            yield '(intercept)', self.classifier.intercept_[0]
            for name, value in zip(self.features, self.classifier.coef_[0]):
                yield name, value
        else:
            logger.warning("Method weights unsupported for %s", self.classname)
            return


class TrainClassifier:
    """Classify clean and noisy sentence pairs"""

    def __init__(self, training_scores=None, dev_scores=None, model_type=None,
                 model_parameters=None, features=None):
        logger.info("Loading training data")
        self.df_training_data = load_dataframe(training_scores)

        self.group_config = features
        self.feature_config = {}
        for t_key in self.df_training_data.keys():
            for f_key in features.keys():
                if t_key.startswith(f_key):
                    self.feature_config[t_key] = features[f_key]

        self.df_training_data = self.df_training_data[self.feature_config.keys()]
        self.df_training_data, self.means_stds = standardize_dataframe_scores(
                    self.df_training_data, self.feature_config)

        if dev_scores:
            logger.info("Loading development data")
            self.dev_data = load_dataframe(dev_scores)
            self.dev_labels = self.dev_data.pop('label')
            self.dev_data = self.dev_data[self.feature_config.keys()]
            self.dev_data = standardize_dataframe_scores(
                    self.dev_data, self.feature_config, self.means_stds)[0]
        else:
            self.dev_data = None
            self.dev_labels = None

        if model_type is None:
            self.model_type = 'LogisticRegression'
        else:
            self.model_type = model_type
        if model_parameters is None:
            self.model_parameters = {}
        else:
            self.model_parameters = model_parameters

    def train_classifier(self, training_data, labels):
        """Train logistic regression with training_data"""
        classifier = Classifier(self.model_type, self.model_parameters,
                                training_data.columns, self.means_stds)
        classifier.train(training_data, labels, standardize=False)
        return classifier

    def get_roc_auc(self, model, dev_data):
        """Calculate ROC AUC for a given model (requires dev_data)"""
        probs = model.classifier.predict_proba(dev_data)
        # pred = model.classifier.predict(dev_data)
        # logger.info("Classifier labels: %s", model.classifier.classes_)
        # logger.info("Predicted labels: %s", collections.Counter(pred))
        return roc_auc_score(self.dev_labels, probs[:, 1])

    @staticmethod
    def get_sse(model, training_data, labels):
        """Calculate the residual sum of squares"""
        y_hat = model.classifier.predict(training_data)
        resid = labels - y_hat
        sse = sum(resid**2)+0.01
        return sse

    @staticmethod
    def get_ce(model, training_data, labels):
        """Calculate cross entropy for a given model"""
        y_pred = model.classifier.predict_proba(training_data)
        return log_loss(labels, y_pred)

    @classmethod
    def get_aic(cls, model, training_data, labels):
        """Calculate AIC for a given model"""
        loss = cls.get_ce(model, training_data, labels)
        k = training_data.shape[1]  # number of variables
        aic = 2 * k - 2 * math.log(loss)
        return aic

    @classmethod
    def get_bic(cls, model, training_data, labels):
        """Calculate BIC for a given model"""
        # pylint: disable=C0103
        loss = cls.get_ce(model, training_data, labels)
        k = training_data.shape[1]  # number of variables
        n = training_data.shape[0]  # number of observations
        bic = n * math.log(loss / n) + k * math.log(n)
        return bic

    @staticmethod
    def get_labels(training_data, cutoffs):
        """Get labels for training data based on cutoffs"""
        labels = []
        training_data_dict = training_data.copy().to_dict()
        for i in range(len(training_data.index)):
            label = 1
            for key in cutoffs.keys():
                if training_data_dict[key][i] < cutoffs[key]:
                    label = 0
            labels.append(label)
        return labels

    @staticmethod
    def get_cutoffs(training_data, quantiles, features):
        """Get cutoff values based on discard percentages"""
        cutoffs = {}
        for key in features:
            cutoffs[key] = training_data[key].quantile(quantiles[key])
        return cutoffs

    @staticmethod
    def _load_feature_bounds_and_init(fdict):
        """Load feature boundaries and initial values from config dict"""
        features = []
        bounds = []
        initial = []
        for key, params in fdict.items():
            features.append(key)
            if 'quantiles' in params:
                min_ = params['quantiles'].get('min', 0)
                max_ = params['quantiles'].get('max', 1)
            else:
                min_, max_ = 0, 1
                logger.warning(
                    "No quantile bounds defined for %s, setting to [%s, %s]",
                    key, min_, max_)
            bounds.append([min_, max_])
            if 'initial' in params.get('quantiles', {}):
                init = params['quantiles']['initial']
            else:
                init = 0.1
                logger.warning(
                    "No initial quantile defined for %s, setting to %s",
                    key, init)
            initial.append(init)
        initial = np.array(initial)
        return features, bounds, initial

    def _cost(self, qvector, features, criterion):
        """Return cost of qvector for given features and criterion"""
        best_quantiles = dict(zip(features, qvector))
        logger.info('Training logistic regression model with quantiles:\n%s',
                    '\n'.join(f'* {t[0]}: {t[1]}' for t in best_quantiles.items()))
        if any(q == 0 for q in best_quantiles.values()):
            # Remove unused features
            df_train_copy = self.df_training_data.copy()
            if self.dev_data is not None:
                df_dev_copy = self.dev_data.copy()
            active = set(features)
            for key, value in best_quantiles.items():
                if value == 0:
                    df_train_copy.pop(key)
                    if self.dev_data is not None:
                        df_dev_copy.pop(key)
                    active.remove(key)
        else:
            df_train_copy = self.df_training_data
            df_dev_copy = self.dev_data
            active = set(features)

        cutoffs = self.get_cutoffs(df_train_copy, best_quantiles, active)
        labels = self.get_labels(df_train_copy, cutoffs)
        counts = collections.Counter(labels)
        logger.info("Label counts in data: %s", counts)
        if len(counts) > 1:
            classifier = self.train_classifier(df_train_copy, labels)
            if criterion['dev']:
                crit_value = criterion['func'](classifier, df_dev_copy)
            else:
                crit_value = criterion['func'](classifier, df_train_copy, labels)
        else:
            crit_value = np.inf if criterion['best'] == 'low' else -np.inf

        logger.info('Model criterion value: %s', crit_value)
        return crit_value if criterion['best'] == 'low' else -crit_value

    def _prune_datasets(self, quantiles, features):
        """Return datasets without features that have zero value in quantiles"""
        df_train_copy = self.df_training_data.copy()
        if self.dev_data is not None:
            df_dev_copy = self.dev_data.copy()
        active = set(features)
        for key, value in quantiles.items():
            if value == 0:
                df_train_copy.pop(key)
                if self.dev_data is not None:
                    df_dev_copy.pop(key)
                active.remove(key)
        return df_train_copy, df_dev_copy, active

    def _get_criterion(self, name):
        """Return function and specifications for optimization criterion"""
        criteria = {
            'AIC': {'func': self.get_aic, 'best': 'low', 'dev': False},
            'BIC': {'func': self.get_bic, 'best': 'low', 'dev': False},
            'SSE': {'func': self.get_sse, 'best': 'low', 'dev': False},
            'CE': {'func': self.get_ce, 'best': 'low', 'dev': False},
            'ROC_AUC': {'func': self.get_roc_auc, 'best': 'high', 'dev': True}
        }
        if name not in criteria:
            raise ValueError(f'Invalid criterion. Expected one of: {", ".join(criteria)}')
        return criteria[name]

    def find_best_model(self, criterion_name, algorithm='default', options=None):
        """Find the model with the best AIC / BIC / SSE / CE / ROC_AUC"""
        criterion = self._get_criterion(criterion_name)
        features, bounds, initial = self._load_feature_bounds_and_init(self.feature_config)
        cost = functools.partial(self._cost, features=features, criterion=criterion)
        if options is None:
            options = {}
        if algorithm == 'none':
            # Use initial values
            best_quantiles = dict(zip(features, initial))
        elif algorithm == 'default':
            # Default local search with multiplicative updates
            best_quantiles = dict(zip(features, self.default_search(cost, initial, bounds=bounds, **options)))
        else:
            # Use optimization algorithm from scipy
            best_quantiles = dict(zip(features, scipy.optimize.minimize(
                cost, initial, method=algorithm, bounds=bounds, options=options).x))
        df_train_copy, df_dev_copy, active = self._prune_datasets(best_quantiles, features)
        labels = self.get_labels(df_train_copy, self.get_cutoffs(df_train_copy, best_quantiles, active))
        classifier = self.train_classifier(df_train_copy, labels)
        if criterion['dev']:
            return classifier, criterion['func'](classifier, df_dev_copy), best_quantiles
        return classifier, criterion['func'](classifier, df_train_copy, labels), best_quantiles

    @staticmethod
    def default_search(costfunc, initial, bounds=None, step_coef=1.25):
        """Local search algorithm with multiplicative updates"""
        if bounds is None:
            bounds = [(0, 1) for _ in range(len(initial))]
        cur_x = initial.copy()
        cur_cost = costfunc(cur_x)
        while True:
            no_change = 0
            for fidx in range(len(initial)):
                new_x = cur_x.copy()
                if new_x[fidx] / step_coef >= bounds[fidx][0]:
                    new_x[fidx] /= step_coef
                    cost = costfunc(new_x)
                    if cost < cur_cost:
                        cur_cost = cost
                        cur_x = new_x
                        continue
                new_x = cur_x.copy()
                if new_x[fidx] * step_coef <= bounds[fidx][1]:
                    new_x[fidx] *= step_coef
                    cost = costfunc(new_x)
                    if cost < cur_cost:
                        cur_cost = cost
                        cur_x = new_x
                        continue
                no_change += 1
            if no_change == len(initial):
                return cur_x
