import os
import requests
import shutil
import tempfile
import unittest

from opusfilter import ConfigurationError
from opusfilter.filters import *
from opusfilter.util import file_download


class TestLongestCommonSubstringFilter(unittest.TestCase):

    bi_inputs = [('abcd', 'abcd'), ('abcd', 'efgh'), ('abcd', 'cdgh'), ('abcd', ''), ('', ''),
                 ('abcd', 'bc'), ('abcd', 'ab abcd cd'), ('abcd ', ' abcd'), ('ab cd', 'a bc d')]
    tri_inputs = [('abcd', 'abcd', 'abcd'), ('abcd', 'abcd', 'efgh'), ('abcd', '', ''), ('', '', ''),
                  ('abcd', 'abc', 'bc'), ('abcd', 'xbcd', 'xabx')]

    def test_bilingual(self):
        testfilter = LongestCommonSubstringFilter(threshold=0.8, require_all=True)
        expected = [([1], False), ([0], True), ([0.5], True), ([0], True), ([0], True),
                    ([1], False), ([1], False), ([0.8], False), ([0.2], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.bi_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_trilingual(self):
        testfilter = LongestCommonSubstringFilter(threshold=0.75, require_all=True)
        expected = [([1, 1, 1], False), ([1, 0, 0], False), ([0, 0, 0], True), ([0, 0, 0], True),
                    ([1, 1, 1], False), ([0.75, 0.5, 0.25], False)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.tri_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_trilingual_any(self):
        testfilter = LongestCommonSubstringFilter(threshold=0.75, require_all=False)
        expected = [([1, 1, 1], False), ([1, 0, 0], True), ([0, 0, 0], True), ([0, 0, 0], True),
                    ([1, 1, 1], False), ([0.75, 0.5, 0.25], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.tri_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)


class TestFasttext(unittest.TestCase):

    fasttext_inputs = ["This sentence is in english", "Je suis une phrase en français"]

    pairs_inputs = [
        ("This sentence is in english", "Je suis une phrase en français"),
        ("me llamo bernardo", "je m'appelle Bernard")
    ]

    model_url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz'

    @classmethod
    def setUpClass(self):
        self.tempdir = tempfile.mkdtemp()
        self.testmodel = os.path.join(self.tempdir, 'model.ftz')
        try:
            file_download(self.model_url, self.testmodel)
        except requests.exceptions.ConnectionError:
            self.testmodel = None

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tempdir)

    def test_missing_model(self):
        with self.assertRaises(ConfigurationError):
            model = LanguageIDFilter(
                languages=['en', 'fr'], id_method='fasttext', thresholds=[0.8, 0.99])

    def test_wrong_method_with_model(self):
        with self.assertRaises(ConfigurationError):
            model = LanguageIDFilter(
                languages=['en', 'fr'], thresholds=[0.8, 0.99], fasttext_model_path=self.tempdir)

    def test_fasttext_predict_lang(self):
        if self.testmodel is None:
            self.skipTest("Failed to download test resources")
        model = LanguageIDFilter(
            languages=['en', 'fr'], id_method='fasttext', thresholds=[0.8, 0.99],
            fasttext_model_path=self.testmodel)
        expected = ['en', 'fr']
        results = [model._fasttext_predict_lang(fasttext_input)[0]
                   for fasttext_input in self.fasttext_inputs]
        self.assertSequenceEqual(expected, results)

    def test_fasttext_accept(self):
        if self.testmodel is None:
            self.skipTest("Failed to download test resources")
        model = LanguageIDFilter(
            languages=['en', 'fr'], id_method='fasttext', thresholds=[0.8, 0.99],
            fasttext_model_path=self.testmodel)
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)
