"""Filter classifier"""

import json
import logging
import collections
import math

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class FilterClassifier:
    """Classify clean and noisy sentence pairs"""

    def __init__(self, training_scores=None, to_be_classified=None,
            output_file=None, dev_scores=None):
        nested_train_data = self.load_data(training_scores)
        training_data = self.unpack_data(nested_train_data)
        self.training_data = pd.DataFrame(training_data)

        nested_tbc_data = self.load_data(to_be_classified)
        tbc_data = self.unpack_data(nested_tbc_data)
        self.tbc_data = pd.DataFrame(tbc_data)

        if output_file:
            self.output_file = output_file
        else:
            self.output_file = to_be_classified+'.probabilities.txt'

        if dev_scores:
            nested_dev_data = self.load_data(dev_scores)
            dev_data = self.unpack_data(nested_dev_data)
            self.dev_data = pd.DataFrame(dev_data)
            self.dev_labels = self.dev_data.pop('label')

    def load_data(self, data_file):
        """Load data from a jsonlines file"""
        data = []
        with open(data_file) as dfile:
            for line in dfile:
                data.append(json.loads(line))
        return data

    def unpack_item(self, item, prev_keys, new_data):
        """Unpack a nested dictionary item into a single item"""
        for key in item.keys():
            if type(item[key]) == dict:
                self.unpack_item(item[key], prev_keys + [key], new_data)
            else:
                new_key = '_'.join(prev_keys+[key])
                if new_key not in new_data:
                    new_data[new_key] = []
                new_data[new_key].append(item[key])
        return new_data

    def unpack_data(self, data):
        """Unpack all items in data"""
        new_data = {}
        for item in data:
            new_data = self.unpack_item(item, [], new_data)
        return new_data

    def set_cutoffs(self, discard, cutoffs):
        """Set cutoff values based on discard percentage"""
        for key in cutoffs.keys():
            if 'CrossEntropyFilter' in key:
                cutoffs[key] = self.training_data[key].quantile(1-discard)
            else:
                cutoffs[key] = self.training_data[key].quantile(discard)
        return cutoffs

    def add_labels(self, cutoffs):
        """Add labels to training data based on cutoffs"""
        labels = []
        for i in range(len(self.training_data.index)):
            label = 1
            for key in cutoffs.keys():
                if 'CrossEntropyFilter' in key:
                    if self.training_data[key][i] > cutoffs[key]:
                        label = 0
                else:
                    if self.training_data[key][i] < cutoffs[key]:
                        label = 0
            labels.append(label)
        self.labels_train= labels

    def train_logreg(self):
        """Train logistic regression with training_data"""
        x_train = self.training_data.to_numpy()
        y_train = self.labels_train
        LR = LogisticRegression(solver='liblinear')
        LR.fit(x_train, y_train)
        return LR

    def get_roc_auc(self, model):
        """Calculate ROC AUC for a given model (requires dev_data)"""
        pred = model.predict(self.dev_data)
        probs = model.predict_proba(self.dev_data)
        score = model.score(self.dev_data, self.dev_labels)
        auc1 = roc_auc_score(self.dev_labels, probs[:,0])
        auc2 = roc_auc_score(self.dev_labels, probs[:,1])
        return max(auc1, auc2)

    def get_sse(self, model):
        """Calculate the residual sum of squares"""
        y_hat = model.predict(self.training_data)
        resid = self.labels_train - y_hat
        sse = sum(resid**2)+0.01
        return sse

    def get_aic(self, model):
        """Calculate AIC for a given model"""
        sse = self.get_sse(model)
        k = self.training_data.shape[1] # number of variables
        AIC = 2*k - 2*math.log(sse)
        return AIC

    def get_bic(self, model):
        """Calculate BIC for a given model"""
        sse = self.get_sse(model)
        k = self.training_data.shape[1] # number of variables
        n = self.training_data.shape[0] # number of observations
        BIC = n*math.log(sse/n) + k*math.log(n)
        return BIC

    def find_best_model(self, discard_thresholds, criterion):
        """Find the model with the best ROC AUC / AIC / BIC"""
        cutoffs = {key: None for key in self.training_data.keys()}
        best = None
        for discard_threshold in discard_thresholds:
            cutoffs = self.set_cutoffs(discard_threshold, cutoffs)
            self.add_labels(cutoffs)
            LR = self.train_logreg()

            if criterion == 'roc_auc':
                crit_value = 1 - self.get_roc_auc(LR) # smaller is better
            elif criterion == 'AIC':
                crit_value = self.get_aic(LR)
            elif criterion == 'BIC':
                crit_value = self.get_bic(LR)

            if best == None or crit_value < best[1]:
                best = (LR, crit_value, discard_threshold)

        return best

    def assign_probabilities(self, model):
        """Assign probabilities to the to_be_classified data"""
        probas = model.predict_proba(self.tbc_data)
        with open(self.output_file, 'w') as output:
            for proba in probas[:,1]:
                output.write(str(proba)+'\n')
