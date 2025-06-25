import logging
import os
import shutil
import tempfile
import unittest

import requests

from opusfilter import ConfigurationError
from opusfilter.lid import *
from opusfilter.util import file_download


try:
    import fasttext
except ImportError:
    logging.warning("Could not import fasttext")

try:
    import pycld2
except ImportError:
    logging.warning("Could not import pycld2")

try:
    import heliport
except ImportError:
    logging.warning("Could not import heliport")


class TestLangIDMethod(unittest.TestCase):

    pairs_inputs = [
        ("This sentence is in english", "Je suis une phrase en français"),
        ("me llamo bernardo", "je m'appelle Bernard")
    ]


class TestLangId(TestLangIDMethod):

    def test_accept(self):
        model = LangidFilter(
            languages=['en', 'fr'], thresholds=[0.8, 0.99])
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)

    def test_accept_with_set_languages(self):
        model = LangidFilter(
            languages=['en', 'fr'], thresholds=[0.8, 0.99],
            langid_languages=['fr', 'de'])
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [False, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)


class TestCLD2(TestLangIDMethod):

    pairs_inputs = [
        ("This sentence is in english", "Je suis une phrase en français"),
        ("me llamo bernardo", "je m'appelle Bernard"),
        ("english sentence", "phrase français")
    ]

    @unittest.skipIf('pycld2' not in globals(), 'pycld2 not installed')
    def test_accept(self):
        model = Cld2Filter(
            languages=['en', 'fr'], thresholds=[0.9, 0.9])
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)

    @unittest.skipIf('pycld2' not in globals(), 'pycld2 not installed')
    def test_accept_with_options(self):
        model = Cld2Filter(
            languages=['en', 'fr'], thresholds=[0.9, 0.9],
            options={'bestEffort': True})
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False, True]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            logging.info('%s %s', pair_score, pair_expected)
            self.assertEqual(model.accept(pair_score), pair_expected)


class TestFasttext(TestLangIDMethod):

    fasttext_inputs = ["This sentence is in english", "Je suis une phrase en français"]
    model_url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz'

    @classmethod
    def setUpClass(self):
        self.tempdir = tempfile.mkdtemp()
        if 'fasttext' not in globals():
            raise unittest.SkipTest('fasttext not installed')
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
            model = FastTextFilter(
                languages=['en', 'fr'], thresholds=[0.8, 0.99])

    def test_fasttext_predict_lang(self):
        if self.testmodel is None:
            self.skipTest("Failed to download test resources")
        model = FastTextFilter(
            languages=['en', 'fr'], thresholds=[0.8, 0.99],
            model_path=self.testmodel)
        expected = ['en', 'fr']
        results = [model._fasttext_predict_lang(fasttext_input)[0]
                   for fasttext_input in self.fasttext_inputs]
        self.assertSequenceEqual(expected, results)

    def test_accept(self):
        if self.testmodel is None:
            self.skipTest("Failed to download test resources")
        model = FastTextFilter(
            languages=['en', 'fr'], thresholds=[0.8, 0.99],
            model_path=self.testmodel)
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)


class TestLingua(TestLangIDMethod):

    def test_accept(self):
        model = LinguaFilter(
            languages=['en', 'fr'], thresholds=[0.4, 0.99])
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)

    def test_accept_high(self):
        model = LinguaFilter(
            languages=['en', 'fr'], lingua_mode="high", thresholds=[0.5, 0.7])
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)

    def test_limited_languages(self):
        model = LinguaFilter(
            languages=['en', 'fr'], thresholds=[0.4, 0.99],
            langid_languages=['en', 'fr', 'de', 'fi'])
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)

    def test_badly_limited_languages(self):
        model = LinguaFilter(
            languages=['en', 'fr'], thresholds=[0.4, 0.99],
            langid_languages=['en', 'de', 'fi'])
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [False, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)


class TestHeliport(TestLangIDMethod):

    @unittest.skipIf('heliport' not in globals(), 'heliport not installed')
    def test_simple_accept(self):
        model = HeliportSimpleFilter(languages=['en', 'fr'], thresholds=[0.2, 0.2])
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)

    @unittest.skipIf('heliport' not in globals(), 'heliport not installed')
    def test_confidence_accept(self):
        model = HeliportConfidenceFilter(languages=['en', 'fr'], thresholds=[0.5, 0.5])
        pair_scores = list(model.score(self.pairs_inputs))
        logging.warning(pair_scores)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)

    @unittest.skipIf('heliport' not in globals(), 'heliport not installed')
    def test_probability_accept(self):
        model = HeliportProbabilityFilter(
            languages=['en', 'fr'], thresholds=[0.3, 0.3], topk=50)
        pair_scores = list(model.score(self.pairs_inputs))
        logging.warning(pair_scores)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)

    @unittest.skipIf('heliport' not in globals(), 'heliport not installed')
    def test_raw_accept(self):
        model = HeliportRawScoreFilter(
            languages=['en', 'fr'], thresholds=[5.0, 5.0], topk=50)
        pair_scores = list(model.score(self.pairs_inputs))
        logging.warning(pair_scores)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)
