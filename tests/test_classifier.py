import unittest
import os
import tempfile
import shutil

import pandas as pd

from opusfilter.classifier import TrainClassifier
from opusfilter.classifier import standardize_dataframe_scores


class TestTrainClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempdir = tempfile.mkdtemp()
        traindata = """{"CharacterScoreFilter": {"src": 1, "tgt": 1}, "LanguageIDFilter": {"cld2": {"src": 1, "tgt": 1}, "langid": {"src": 1, "tgt": 1}}, "LongWordFilter": 1}
        {"CharacterScoreFilter": {"src": 2, "tgt": 2}, "LanguageIDFilter": {"cld2": {"src": 2, "tgt": 2}, "langid": {"src": 2, "tgt": 2}}, "LongWordFilter": 2}
        {"CharacterScoreFilter": {"src": 3, "tgt": 3}, "LanguageIDFilter": {"cld2": {"src": 3, "tgt": 3}, "langid": {"src": 3, "tgt": 3}}, "LongWordFilter": 3}
        {"CharacterScoreFilter": {"src": 4, "tgt": 4}, "LanguageIDFilter": {"cld2": {"src": 4, "tgt": 4}, "langid": {"src": 4, "tgt": 4}}, "LongWordFilter": 4}
        {"CharacterScoreFilter": {"src": 5, "tgt": 5}, "LanguageIDFilter": {"cld2": {"src": 5, "tgt": 5}, "langid": {"src": 5, "tgt": 5}}, "LongWordFilter": 5}"""
        with open(os.path.join(self.tempdir, 'scores.jsonl'), 'w') as f:
            f.write(traindata)
            self.jsonl_train = os.path.join(self.tempdir, 'scores.jsonl')

        devdata = """{"CharacterScoreFilter": {"src": 4, "tgt": 4}, "LanguageIDFilter": {"cld2": {"src": 4, "tgt": 4}, "langid": {"src": 4, "tgt": 4}}, "LongWordFilter": 4, "label": 0}
        {"CharacterScoreFilter": {"src": 5, "tgt": 5}, "LanguageIDFilter": {"cld2": {"src": 5, "tgt": 5}, "langid": {"src": 5, "tgt": 5}}, "LongWordFilter": 5, "label": 1}"""

        with open(os.path.join(self.tempdir, 'dev.jsonl'), 'w') as f:
            f.write(devdata)
            self.jsonl_dev = os.path.join(self.tempdir, 'dev.jsonl')


        features = {'CharacterScoreFilter':
                    {'clean-direction': 'high',
                        'quantiles': [0.1, 0.09, 0.08, 0.07, 0.06, 0.05,
                            0.04, 0.03, 0.02, 0.01, 0.0]},
                'LanguageIDFilter':
                    {'clean-direction': 'high',
                        'quantiles': [0.1, 0.09, 0.08, 0.07, 0.06, 0.05,
                            0.04, 0.03, 0.02, 0.01]},
                'LongWordFilter':
                    {'clean-direction': 'high',
                        'quantiles': [0.1, 0.09, 0.08, 0.07, 0.06, 0.05,
                            0.04, 0.03, 0.02, 0.01]}
                }

        self.fc = TrainClassifier(
                training_scores=self.jsonl_train,
                features = features,
                dev_scores=self.jsonl_dev)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tempdir)

    def test_set_cutoffs(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.5 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        self.assertEqual(new_cutoffs['LongWordFilter'], 0.0)
        discards = {key: 0.25 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        self.assertEqual(new_cutoffs['LanguageIDFilter.cld2.src'],
                -0.7071067811865475)
        discards = {key: 0.75 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        self.assertEqual(new_cutoffs['LanguageIDFilter.cld2.tgt'],
                0.7071067811865475)

    def test_add_labels(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.26 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        self.fc.add_labels(self.fc.df_training_data, new_cutoffs)
        ones = sum(self.fc.labels_train)
        self.assertEqual(ones, 3)
        discards = {key: 0.25 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        self.fc.add_labels(self.fc.df_training_data, new_cutoffs)
        ones = sum(self.fc.labels_train)
        self.assertEqual(ones, 4)

    def test_train_logreg(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.26 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        labels = self.fc.add_labels(self.fc.df_training_data, new_cutoffs)
        LR = self.fc.train_logreg(self.fc.df_training_data, labels)
        self.assertAlmostEqual(round(LR.classifier.intercept_[0], 8),
                0.30343394)

    def test_get_roc_auc(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.26 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        labels = self.fc.add_labels(self.fc.df_training_data, new_cutoffs)
        LR = self.fc.train_logreg(self.fc.df_training_data, labels)
        self.assertAlmostEqual(self.fc.get_roc_auc(LR, self.fc.dev_data), 1)

    def test_get_aic(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.26 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        labels = self.fc.add_labels(self.fc.df_training_data, new_cutoffs)
        LR = self.fc.train_logreg(self.fc.df_training_data, labels)
        aic = self.fc.get_aic(LR, self.fc.df_training_data, labels)
        self.assertAlmostEqual(aic, 19.21034037197618)

    def test_get_bic(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.26 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        labels = self.fc.add_labels(self.fc.df_training_data, new_cutoffs)
        LR = self.fc.train_logreg(self.fc.df_training_data, labels)
        bic = self.fc.get_bic(LR, self.fc.df_training_data, labels)
        self.assertAlmostEqual(bic, 17.25752993414668)

    def test_find_best_roc_auc_model(self):
        LR, roc_auc, value = self.fc.find_best_model('roc_auc')
        self.assertAlmostEqual(roc_auc, 1)

    def test_standardize_dataframe_scores(self):
        data = {'column1': [3,4,3,6,3,7,3],
                'column2': [4,8,6,9,2,-5,4],
                'column3': [1,11,3,7,4,-1,6],
                'column4': [0,1,2,3,4,5,6],
                'column5': [1,1,1,1,1,1,1]}
        features = {'column1': {'clean-direction': 'high', 'quantiles': [0.1]},
                'column2': {'clean-direction': 'high', 'quantiles': [0.1]},
                'column3': {'clean-direction': 'high', 'quantiles': [0.1]},
                'column4': {'clean-direction': 'high', 'quantiles': [0.1]},
                'column5': {'clean-direction': 'high', 'quantiles': [0.1]}}
        df = pd.DataFrame(data)
        new_df, means_stds = standardize_dataframe_scores(df, features)
        self.assertEqual(df['column1'][0], 3)
        self.assertEqual(df['column4'][6], 6)
        self.assertAlmostEqual(round(new_df['column1'][0], 6), -0.73646)
        self.assertAlmostEqual(round(new_df['column4'][6], 6), 1.5)
        self.assertAlmostEqual(new_df['column5'][6], 0)

    def test_find_best_aic_model(self):
        pass
        #LR, aic, value, weights = self.fc.find_best_model('AIC')
        #self.assertAlmostEqual(aic, 13.980099338293664)

    def test_find_best_bic_model(self):
        pass
        #LR, bic, value, weights = self.fc.find_best_model('BIC')
        #self.assertAlmostEqual(bic, 11.246164725332367)

    def test_assign_scores(self):
        pass
        #LR, roc_auc, value, weights = self.fc.find_best_model('roc_auc')
        #probas = self.fc.assign_probabilities(LR)
        #with open(self.fc.output_file) as output:
        #    lines = output.readlines()
        #self.assertAlmostEqual(float(lines[0]), 0.6444675902763973)
        #self.assertAlmostEqual(float(lines[-1]), 0.9873019540450083)

