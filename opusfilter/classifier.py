"""Filter classifier"""

import json
import logging
import collections

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class FilterClassifier:
    """Classify clean and noisy sentence pairs"""

    def __init__(self, training_scores, dev_scores=None):
        nested_data = self.load_data(training_scores)
        training_data = self.unpack_data(nested_data)
        self.training_data = pd.DataFrame(training_data)
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
        """Calculate ROC AUC for a given model"""
        pred = model.predict(self.dev_data)
        probs = model.predict_proba(self.dev_data)
        score = model.score(self.dev_data, self.dev_labels)
        auc1 = roc_auc_score(self.dev_labels, probs[:,0])
        auc2 = roc_auc_score(self.dev_labels, probs[:,1])
        return max(auc1, auc2)

    def find_best_roc_auc(self, discard_thresholds):
        """Find the model with the best ROC AUC based on discard thresholds"""
        cutoffs = {key: None for key in self.training_data.keys()}
        best = None
        for value in discard_thresholds:
            cutoffs = self.set_cutoffs(value, cutoffs)
            self.add_labels(cutoffs)
            LR = self.train_logreg()
            roc_auc = self.get_roc_auc(LR)
            if best == None or roc_auc > best[1]:
                best = (LR, roc_auc, value)
        return best

