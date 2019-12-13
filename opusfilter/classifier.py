"""Filter classifier"""

import json
import logging
import collections
import math

import pandas as pd
from pandas.io.json import json_normalize
import sklearn.linear_model
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def load_dataframe(data_file):
    """Load normalized scores dataframe from a jsonlines file"""
    data = []
    with open(data_file) as dfile:
        for line in dfile:
            data.append(json.loads(line))
    return pd.DataFrame(json_normalize(data))

def standardize_dataframe_scores(df, features, means_stds=None):
    """Normalize and zero average scores in each column"""
    new_df = pd.DataFrame()
    if not means_stds:
        means_stds = {}
        for column in df:
            x = df[column].to_numpy()
            means_stds[column] = (x.mean(), x.std())
    for column in df:
        x = df[column].to_numpy()
        mean, std = means_stds[column]
        if std == 0:
            x = 0
        else:
            x = (x-mean)/std
        if features[column]['clean-direction'] == 'low':
            x = x*-1
        new_df[column] = x
    return new_df, means_stds


class Classifier:

    def __init__(self, classname, params, features):
        self.classname = classname
        cls = getattr(sklearn.linear_model, self.classname)
        self.classifier = cls(**params)
        self.features = features

    def train(self, df, labels):
        """Train logistic regression with training_data"""
        self.classifier.fit(df[self.features], labels)

    def write_preds(self, input_fname, output_fname):
        """Write predicted class labels to output file"""
        df_tbc = load_dataframe(input_fname)
        labels = self.classifier.predict(df_tbc[self.features])
        with open(output_fname, 'w') as output:
            for label in labels:
                output.write('{}\n'.format(label))

    def write_probs(self, input_fname, output_fname):
        """Write classification probabilities to output file"""
        df_tbc = load_dataframe(input_fname)
        probas = self.classifier.predict_proba(df_tbc[self.features])
        with open(output_fname, 'w') as output:
            for proba in probas[:,1]:
                output.write('{0:.10f}\n'.format(proba))

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
            model_parameters=None, features=None, **kwargs):
        self.df_training_data = load_dataframe(training_scores)

        self.features = {}
        for t_key in self.df_training_data.keys():
            for f_key in features.keys():
                if t_key.startswith(f_key):
                    self.features[t_key] = features[f_key]

        self.df_training_data = self.df_training_data[self.features.keys()]
        self.df_training_data, self.means_stds = standardize_dataframe_scores(
                    self.df_training_data, self.features)

        if dev_scores:
            self.dev_data = load_dataframe(dev_scores)
            self.dev_labels = self.dev_data.pop('label')
            self.dev_data = self.dev_data[self.features.keys()]
            self.dev_data = standardize_dataframe_scores(
                    self.dev_data, self.features, self.means_stds)[0]

        if model_type == None:
            self.model_type = 'LogisticRegression'
        else:
            self.model_type = model_type
        if model_parameters == None:
            self.model_parameters = {}
        else:
            self.model_parameters = model_parameters

    def train_logreg(self, training_data, labels):
        """Train logistic regression with training_data"""
        x_train = training_data.to_numpy()
        y_train = labels
        classifier = Classifier(self.model_type, self.model_parameters,
                training_data.keys())
        classifier.train(training_data, labels)
        return classifier

    def get_roc_auc(self, model, dev_data):
        """Calculate ROC AUC for a given model (requires dev_data)"""
        pred = model.classifier.predict(dev_data)
        probs = model.classifier.predict_proba(dev_data)
        score = model.classifier.score(dev_data, self.dev_labels)
        auc1 = roc_auc_score(self.dev_labels, probs[:,0])
        auc2 = roc_auc_score(self.dev_labels, probs[:,1])
        return max(auc1, auc2)

    def get_sse(self, model, training_data, labels):
        """Calculate the residual sum of squares"""
        y_hat = model.classifier.predict(training_data)
        resid = self.labels_train - y_hat
        sse = sum(resid**2)+0.01
        return sse

    def get_aic(self, model, training_data, labels):
        """Calculate AIC for a given model"""
        sse = self.get_sse(model, training_data, labels)
        k = training_data.shape[1] # number of variables
        AIC = 2*k - 2*math.log(sse)
        return AIC

    def get_bic(self, model, training_data, labels):
        """Calculate BIC for a given model"""
        sse = self.get_sse(model, training_data, labels)
        k = training_data.shape[1] # number of variables
        n = training_data.shape[0] # number of observations
        #BIC = n*math.log(sse/n) + k*math.log(n)
        BIC = math.log(n)*k - 2*math.log(sse)
        return BIC

    def add_labels(self, training_data, cutoffs):
        """Add labels to training data based on cutoffs"""
        labels = []
        training_data_dict = training_data.copy().to_dict()
        for i in range(len(training_data.index)):
            label = 1
            for key in cutoffs.keys():
                if training_data_dict[key][i] < cutoffs[key]:
                    label = 0
            labels.append(label)
        self.labels_train = labels
        return labels

    def set_cutoffs(self, training_data, discards, cutoffs):
        """Set cutoff values based on discard percentage"""
        for key in cutoffs.keys():
            cutoffs[key] = training_data[key].quantile(discards[key])
        return cutoffs

    def find_best_model(self, criterion):
        """Find the model with the best ROC AUC / AIC / BIC"""
        cutoffs = {key: None for key in self.features.keys()}
        discards = {key: self.features[key]['quantiles'] for key in
                self.features.keys()}
        best_discards = {key: discards[key][0] for key in self.features.keys()}

        best_model = None

        for key in discards.keys():
            quantiles = discards[key]
            best_discard = quantiles[0]
            remove_column = False
            for quantile in quantiles:
                best_discards[key] = quantile
                df_train_copy = self.df_training_data.copy()
                df_dev_copy = self.dev_data.copy()
                cutoffs_copy = cutoffs.copy()

                logger.info('Training logistic regression model with discards'
                    ' {}'.format(best_discards.values()))
                zero = False

                if quantile == 0:
                    zero = True
                    df_train_copy.pop(key)
                    df_dev_copy.pop(key)
                    cutoffs_copy.pop(key)
                else:
                    cutoffs_copy = self.set_cutoffs(df_train_copy,
                            best_discards, cutoffs_copy)
                    labels = self.add_labels(df_train_copy, cutoffs_copy)

                LR = self.train_logreg(df_train_copy, labels)

                if criterion == 'roc_auc':
                    crit_value = self.get_roc_auc(LR, df_dev_copy)
                elif criterion == 'AIC':
                    crit_value = self.get_aic(LR, df_train_copy, labels)
                elif criterion == 'BIC':
                    crit_value = self.get_bic(LR, df_train_copy, labels)

                logger.info('Model {crit}: {value}'.format(
                    crit=criterion, value=crit_value))

                if criterion == 'roc_auc':
                    if best_model == None or crit_value >= best_model[1]:
                        best_model = (LR, crit_value, best_discards)
                        best_discard = quantile
                        if zero:
                            remove_column = True
                elif criterion in ['AIC', 'BIC']:
                    if best_model == None or crit_value < best_model[1]:
                        best_model = (LR, crit_value, best_discards)
                        best_discard = quantile
                        if zero:
                            remove_column = True

            if remove_column:
                self.df_training_data.pop(key)
                self.dev_data.pop(key)
                cutoffs.pop(key)

            best_discards[key] = best_discard

        return best_model

