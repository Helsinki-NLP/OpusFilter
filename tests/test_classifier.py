import unittest
import os
import tempfile
import shutil

from opusfilter.classifier import FilterClassifier


class TestFilterClassifier(unittest.TestCase):

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

        self.fc = FilterClassifier(
                training_scores=self.jsonl_train,
                to_be_classified=self.jsonl_train,
                discard_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
                output_file=os.path.join(self.tempdir, 'output.txt'),
                dev_scores=self.jsonl_dev)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tempdir)

    def test_load_data(self):
        data = self.fc.load_data(self.jsonl_train)
        self.assertEqual(len(data), 5)
        self.assertEqual(type(data[0]), dict)

    def test_unpack_item(self):
        item = {"CharacterScoreFilter": {"src": 1, "tgt": 1},
                "LanguageIDFilter": {"cld2": {"src": 1, "tgt": 1},
                    "langid": {"src": 1, "tgt": 1}},
                "LongWordFilter": 1}
        new_data = {}
        self.fc.unpack_item(item, [], new_data)
        for item in new_data.items():
            self.assertEqual(len(item[1]), 1)

    def test_unpack_data(self):
        for item in self.fc.training_data.items():
            self.assertEqual(len(item[1]), 5)

    def test_set_cutoffs(self):
        cutoffs = {key: None for key in self.fc.training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(0.5, cutoffs)
        self.assertEqual(new_cutoffs['LongWordFilter'], 3)
        new_cutoffs = self.fc.set_cutoffs(0.25, cutoffs)
        self.assertEqual(new_cutoffs['LanguageIDFilter_cld2_src'], 2)
        new_cutoffs = self.fc.set_cutoffs(0.75, cutoffs)
        self.assertEqual(new_cutoffs['CharacterScoreFilter_src'], 4)

    def test_add_labels(self):
        cutoffs = {key: None for key in self.fc.training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(0.26, cutoffs)
        self.fc.add_labels(new_cutoffs)
        ones = sum(self.fc.labels_train)
        self.assertEqual(ones, 3)
        new_cutoffs = self.fc.set_cutoffs(0.25, cutoffs)
        self.fc.add_labels(new_cutoffs)
        ones = sum(self.fc.labels_train)
        self.assertEqual(ones, 4)

    def test_train_logreg(self):
        cutoffs = {key: None for key in self.fc.training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(0.26, cutoffs)
        self.fc.add_labels(new_cutoffs)
        LR = self.fc.train_logreg()
        self.assertAlmostEqual(round(LR.intercept_[0], 8), -0.62285208)

    def test_get_roc_auc(self):
        cutoffs = {key: None for key in self.fc.training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(0.26, cutoffs)
        self.fc.add_labels(new_cutoffs)
        LR = self.fc.train_logreg()
        self.assertAlmostEqual(self.fc.get_roc_auc(LR), 1)

    def test_get_aic(self):
        cutoffs = {key: None for key in self.fc.training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(0.26, cutoffs)
        self.fc.add_labels(new_cutoffs)
        LR = self.fc.train_logreg()
        aic = self.fc.get_aic(LR)
        self.assertAlmostEqual(aic, 13.980099338293664)

    def test_get_bic(self):
        cutoffs = {key: None for key in self.fc.training_data.keys()}
        new_cutoffs = self.fc.set_cutoffs(0.26, cutoffs)
        self.fc.add_labels(new_cutoffs)
        LR = self.fc.train_logreg()
        bic = self.fc.get_bic(LR)
        self.assertAlmostEqual(bic, 11.246164725332367)

    def test_find_best_roc_auc_model(self):
        LR, roc_auc, value = self.fc.find_best_model('roc_auc')
        self.assertAlmostEqual(roc_auc, 1)

    def test_find_best_aic_model(self):
        LR, aic, value = self.fc.find_best_model('AIC')
        self.assertAlmostEqual(aic, 13.980099338293664)

    def test_find_best_bic_model(self):
        LR, bic, value = self.fc.find_best_model('BIC')
        self.assertAlmostEqual(bic, 11.246164725332367)

    def test_assign_scores(self):
        LR, roc_auc, value = self.fc.find_best_model('roc_auc')
        probas = self.fc.assign_probabilities(LR)
        with open(self.fc.output_file) as output:
            lines = output.readlines()
        self.assertAlmostEqual(float(lines[0]), 0.6444675902763973)
        self.assertAlmostEqual(float(lines[-1]), 0.9873019540450083)
