import doctest
import logging
import os
import shutil
import tempfile
import unittest

import json
import pandas as pd

from opusfilter.classifier import *


example_data = [
    {"CharacterScoreFilter": [1, 1], "LanguageIDFilter": {"cld2": [1, 1], "langid": [1, 1]}, "LongWordFilter": 1},
    {"CharacterScoreFilter": [2, 2], "LanguageIDFilter": {"cld2": [2, 2], "langid": [2, 2]}, "LongWordFilter": 2},
    {"CharacterScoreFilter": [3, 3], "LanguageIDFilter": {"cld2": [3, 3], "langid": [3, 3]}, "LongWordFilter": 3},
    {"CharacterScoreFilter": [4, 4], "LanguageIDFilter": {"cld2": [4, 4], "langid": [4, 4]}, "LongWordFilter": 4},
    {"CharacterScoreFilter": [5, 5], "LanguageIDFilter": {"cld2": [5, 5], "langid": [5, 5]}, "LongWordFilter": 5}
]

example_labeled_data = [
    {"CharacterScoreFilter": [4, 4], "LanguageIDFilter": {"cld2": [4, 4], "langid": [4, 4]}, "LongWordFilter": 4, "label": 0},
    {"CharacterScoreFilter": [5, 5], "LanguageIDFilter": {"cld2": [5, 5], "langid": [5, 5]}, "LongWordFilter": 5, "label": 1}
]


class DataframeTests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempdir = tempfile.mkdtemp()
        with open(os.path.join(self.tempdir, 'scores.jsonl'), 'w') as f:
            for item in example_data:
                f.write(json.dumps(item) + '\n')
        self.jsonl_data = os.path.join(self.tempdir, 'scores.jsonl')

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tempdir)
    
    def test_load_dataframe(self):
        df = load_dataframe(self.jsonl_data)
        logging.info(df)
        self.assertEqual(len(df), 5)

    def test_load_dataframe_in_chunks(self):
        dfs = list(load_dataframe_in_chunks(self.jsonl_data, 2))
        self.assertEqual(len(dfs), 3)
        self.assertEqual(len(dfs[0]), 2)
        self.assertEqual(len(dfs[1]), 2)
        self.assertEqual(len(dfs[2]), 1)

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
        self.assertAlmostEqual(new_df['column1'][0], -0.73646, places=6)
        self.assertAlmostEqual(new_df['column4'][6], 1.5, places=6)
        self.assertAlmostEqual(new_df['column5'][6], 0, places=6)


class TestTrainClassifierNoDev(unittest.TestCase):

    features = {
        'CharacterScoreFilter': {
            'clean-direction': 'high',
            'quantiles': {'min': 0, 'max': 0.1, 'initial': 0.02}},
        'LanguageIDFilter': {
            'clean-direction': 'high',
            'quantiles': {'min': 0, 'max': 0.1, 'initial': 0.02}},
        'LongWordFilter': {
            'clean-direction': 'high',
            'quantiles': {'min': 0, 'max': 0.1, 'initial': 0.02}}
    }

    @classmethod
    def setUpClass(self):
        self.tempdir = tempfile.mkdtemp()
        with open(os.path.join(self.tempdir, 'scores.jsonl'), 'w') as f:
            for item in example_data:
                f.write(json.dumps(item) + '\n')
        self.jsonl_train = os.path.join(self.tempdir, 'scores.jsonl')
        self.fc = TrainClassifier(
            training_scores=self.jsonl_train,
            features=self.features,
            model_type='LogisticRegression',
            model_parameters={'solver': 'liblinear'})

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tempdir)

    def test_set_cutoffs(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.5 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.get_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        self.assertEqual(new_cutoffs['LongWordFilter'], 0.0)
        discards = {key: 0.25 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.get_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        self.assertEqual(new_cutoffs['LanguageIDFilter.cld2.0'],
                -0.7071067811865475)
        discards = {key: 0.75 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.get_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        self.assertEqual(new_cutoffs['LanguageIDFilter.cld2.1'],
                0.7071067811865475)

    def test_add_labels(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.26 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.get_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        temp_labels = self.fc.get_labels(self.fc.df_training_data, new_cutoffs)
        ones = sum(temp_labels)
        self.assertEqual(ones, 3)
        discards = {key: 0.25 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.get_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        temp_labels = self.fc.get_labels(self.fc.df_training_data, new_cutoffs)
        ones = sum(temp_labels)
        self.assertEqual(ones, 4)

    def test_train_classifier(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.26 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.get_cutoffs(self.fc.df_training_data, discards, cutoffs)
        labels = self.fc.get_labels(self.fc.df_training_data, new_cutoffs)
        LR = self.fc.train_classifier(self.fc.df_training_data, labels)
        self.assertAlmostEqual(LR.classifier.intercept_[0], 0.31536332, places=8)
        feature_names = set(name for name, weight in LR.weights())
        self.assertSetEqual({
            '(intercept)', 'LongWordFilter',
            'CharacterScoreFilter.0', 'CharacterScoreFilter.1',
            'LanguageIDFilter.langid.0', 'LanguageIDFilter.langid.1',
            'LanguageIDFilter.cld2.0', 'LanguageIDFilter.cld2.1'}, feature_names)
        with tempfile.NamedTemporaryFile(mode='w+') as outfile:
            LR.write_preds(self.jsonl_train, outfile.name)
            outfile.seek(0)
            output = outfile.readlines()
            self.assertEqual(len(output), 5)
        with tempfile.NamedTemporaryFile(mode='w+') as outfile:
            LR.write_probs(self.jsonl_train, outfile.name)
            outfile.seek(0)
            output = outfile.readlines()
            self.assertEqual(len(output), 5)

    def test_get_aic(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.26 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.get_cutoffs(self.fc.df_training_data, discards, cutoffs)
        labels = self.fc.get_labels(self.fc.df_training_data, new_cutoffs)
        LR = self.fc.train_classifier(self.fc.df_training_data, labels)
        aic = self.fc.get_aic(LR, self.fc.df_training_data, labels)
        self.assertAlmostEqual(aic, 17.25174505066196)

    def test_get_bic(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.26 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.get_cutoffs(self.fc.df_training_data, discards, cutoffs)
        labels = self.fc.get_labels(self.fc.df_training_data, new_cutoffs)
        LR = self.fc.train_classifier(self.fc.df_training_data, labels)
        bic = self.fc.get_bic(LR, self.fc.df_training_data, labels)
        self.assertAlmostEqual(bic, -4.910486801786702)

    def test_find_best_aic_model(self):
        LR, aic, value = self.fc.find_best_model('AIC')
        self.assertAlmostEqual(aic, 16.2056, places=2)

    def test_find_best_aic_model_bfgs(self):
        LR, aic, value = self.fc.find_best_model('AIC', algorithm='Powell')
        self.assertAlmostEqual(aic, 16.2056, places=2)

    def test_find_best_aic_model_none(self):
        LR, aic, value = self.fc.find_best_model('AIC', algorithm='none')
        self.assertAlmostEqual(aic, 16.2056, places=2)

    def test_find_best_bic_model(self):
        LR, bic, value = self.fc.find_best_model('BIC')
        self.assertAlmostEqual(bic, -2.295, places=2)

    def test_find_best_sse_model(self):
        LR, sse, value = self.fc.find_best_model('SSE')
        self.assertAlmostEqual(sse, 1.01, places=2)

    def test_find_best_ce_model(self):
        LR, ce, value = self.fc.find_best_model('CE')
        self.assertAlmostEqual(ce, 0.3319, places=2)


class TestTrainClassifierWithDev(TestTrainClassifierNoDev):

    @classmethod
    def setUpClass(self):
        self.tempdir = tempfile.mkdtemp()
        with open(os.path.join(self.tempdir, 'scores.jsonl'), 'w') as f:
            for item in example_data:
                f.write(json.dumps(item) + '\n')
            self.jsonl_train = os.path.join(self.tempdir, 'scores.jsonl')
        with open(os.path.join(self.tempdir, 'dev.jsonl'), 'w') as f:
            for item in example_labeled_data:
                f.write(json.dumps(item) + '\n')
            self.jsonl_dev = os.path.join(self.tempdir, 'dev.jsonl')
        self.fc = TrainClassifier(
                training_scores=self.jsonl_train,
                features=self.features,
                model_type='LogisticRegression',
                model_parameters={'solver': 'liblinear'},
                dev_scores=self.jsonl_dev)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tempdir)

    def test_get_roc_auc(self):
        cutoffs = {key: None for key in self.fc.df_training_data.keys()}
        discards = {key: 0.26 for key in self.fc.df_training_data.keys()}
        new_cutoffs = self.fc.get_cutoffs(self.fc.df_training_data, discards,
                cutoffs)
        labels = self.fc.get_labels(self.fc.df_training_data, new_cutoffs)
        LR = self.fc.train_classifier(self.fc.df_training_data, labels)
        self.assertAlmostEqual(self.fc.get_roc_auc(LR, self.fc.dev_data), 1)

    def test_find_best_roc_auc_model(self):
        LR, roc_auc, value = self.fc.find_best_model('ROC_AUC')
        self.assertAlmostEqual(roc_auc, 1)
