import copy
import unittest

from opusfilter.filters import *


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

    fasttext_model = LanguageIDFilter(languages=["en", "fr"], id_method="fasttext", thresholds=[0.8, 0.99],
                                      fasttext_model_path="./tests/models/lid.176.ftz")

    def test__fasttext_predict_lang(self):
        expected = ['en', 'fr']
        results = [self.fasttext_model._fasttext_predict_lang(fasttext_input)[0] for fasttext_input in self.fasttext_inputs]
        assert expected == results

    def test_fasttext_accept(self):
        pair_scores = self.fasttext_model.score(self.pairs_inputs)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            assert self.fasttext_model.accept(pair_score) == pair_expected
