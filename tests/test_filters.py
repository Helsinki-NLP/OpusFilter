import logging
import os
import requests
import shutil
import tempfile
import unittest

from opusfilter import ConfigurationError
from opusfilter.filters import *
from opusfilter.util import file_download


class TestLengthFilter(unittest.TestCase):

    def test_words(self):
        testfilter = LengthFilter(2, 3, 'word')
        cases = [['a'], ['a dog'], ['a dog went out']]
        expected = [([1], False), ([2], True), ([4], False)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_chars(self):
        testfilter = LengthFilter(4, 10, 'character')
        cases = [['a'], ['a dog'], ['a dog went out']]
        expected = [([1], False), ([5], True), ([14], False)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_chars_bilingual(self):
        testfilter = LengthFilter([6, 8], [18, 15], 'char')
        cases = [['table', 'pyödällä'], ['kitchen table', 'keittiöpöydällä'], ['on the kitchen table', 'keittiöpöydällä']]
        expected = [([5, 8], False), ([13, 15], True), ([20, 15], False)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_mixed_bilingual(self):
        testfilter = LengthFilter([2, 8], [4, 15], ['word', 'char'])
        cases = [['table', 'pyödällä'], ['kitchen table', 'keittiöpöydällä'], ['on the kitchen table', 'keittiöpöydällä']]
        expected = [([1, 8], False), ([2, 15], True), ([4, 15], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)


class TestLengthRatioFilter(unittest.TestCase):

    def test_chars_bilingual(self):
        testfilter = LengthRatioFilter(2, 'char')
        cases = [['table', 'keittiöpyödällä'], ['kitchen table', 'keittiöpöydällä'], ['on the kitchen table', 'keittiöpöydällä']]
        expected = [(len(cases[0][1]) / len(cases[0][0]), False),
                    (len(cases[1][1]) / len(cases[1][0]), True),
                    (len(cases[2][0]) / len(cases[2][1]), True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_mixed_bilingual(self):
        testfilter = LengthRatioFilter(2, ['word', 'char'])
        cases = [['table', '桌'], ['table', '厨房的桌子'], ['kitchen table', '厨房的桌子'], ['on the kitchen table', '在厨房的桌子上']]
        expected = [(1, True), (5, False), (2.5, False), (7 / 4, True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)


class TestLongWordFilter(unittest.TestCase):

    def test_bilingual(self):
        testfilter = LongWordFilter(10)
        cases = [['a bbbbb bbbbbbb', 'c d'], ['aa bbbbbbbbbb', 'c dd e'], ['a bbb aa', 'c ddddddddddd ee'], ['', '']]
        expected = [([7, 1], True), ([10, 2], False), ([3, 11], False), ([0, 0], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)


class TestAverageWordLengthFilter(unittest.TestCase):

    def test_bilingual(self):
        testfilter = AverageWordLengthFilter(1.2, 4)
        cases = [['aa bb bb', 'c d'], ['aa bb', 'c dd e ff'], ['a bbb aa', 'cc ddddddddddd ee'], ['', '']]
        expected = [([2.0, 1.0], False), ([2.0, 1.5], True), ([2.0, 5.0], False), ([0.0, 0.0], False)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)


class TestHtmlTagFilter(unittest.TestCase):

    def test_bilingual(self):
        testfilter = HtmlTagFilter()
        cases = [['aaa bee', 'cee dee'], ['aa <br> bee', 'cee deee'],
                 ['<p>aaa bee</p>', '<p>cee dee</p>'], ['', '']]
        expected = [([False, False], True), ([True, False], False),
                    ([True, True], False), ([False, False], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_broken(self):
        testfilter = HtmlTagFilter()
        cases = [['<aaa bee', 'cee dee'], ['aa br> bee', 'cee deee'],
                 ['<p aaa bee</p', 'p>cee dee /p>'], ['', ''],
                 ['<![ foo', 'foo']]
        expected = [([False, False], True), ([False, False], True),
                    ([False, False], True), ([False, False], True),
                    ([True, False], False)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        logging.warning(results)
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)


class TestRegExpFilter(unittest.TestCase):

    def test_bilingual(self):
        testfilter = RegExpFilter(regexps=['[0-9]', 'a^'], accept_match=False)
        cases = [['aaa', 'bbbb'], ['123', 'bbb'], ['hi123!!!', 'hey...'], ['', '']]
        expected = [([False, False], True), ([True, False], False), ([True, False], False), ([False, False], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_accept_match(self):
        testfilter = RegExpFilter(regexps='^[ a-z]*$', accept_match=True)
        cases = [['aaa'], ['123'], ['hey...'], ['']]
        expected = [([True], True), ([False], False), ([False], False), ([True], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)


class TestAlphabetRatioFilter(unittest.TestCase):

    def test_bilingual(self):
        testfilter = AlphabetRatioFilter(threshold=[0.5, 0.5])
        cases = [['aaa', 'bbbb'], ['123', 'bbb'], ['hi!!!', 'hey...'], [' a  ', 'b '], ['', '']]
        expected = [([1, 1], True), ([0, 1], False), ([0.4, 0.5], False), ([0.25, 0.5], False), ([1, 1], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_exclude_whitespace(self):
        testfilter = AlphabetRatioFilter(threshold=0.5, exclude_whitespace=True)
        cases = [['a    aa'], ['123 '], ['hi !!!'], [' ']]
        expected = [([1], True), ([0], False), ([0.4], False), ([1], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(cases)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)


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


class TestSimilarityFilter(unittest.TestCase):

    bi_inputs = [
        ('abcd', 'abcd'),
        ('abcd', 'efgh'),
        ('abcd', 'ABCD'),
        ('big hat.', 'big hat.'),
        ('big hat.', 'Big Hat.'),
        ('big hat.', 'pig cat.'),
        ('big hat.', 'hat big.'),
    ]
    tri_inputs = [
        ('abcd', 'abcd', 'abcd'),
        ('abcd', 'abcd', 'efgh'),
        ('abcd', 'efgh', 'ijkl')
    ]

    def test_bilingual(self):
        testfilter = SimilarityFilter(threshold=0.7)
        expected = [([1], False), ([0], True), ([0.0], True),
                    ([1], False), ([0.75], False), ([0.75], False), ([0.25], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.bi_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_bilingual_lowercase(self):
        testfilter = SimilarityFilter(threshold=0.7, lowercase=True)
        expected = [([1], False), ([0], True), ([1.0], False),
                    ([1], False), ([1], False), ([0.75], False), ([0.25], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.bi_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_bilingual_word(self):
        testfilter = SimilarityFilter(threshold=0.7, unit='word')
        expected = [([1], False), ([0], True), ([0], True),
                    ([1], False), ([0], True), ([0], True), ([0], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.bi_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_bilingual_word_lowercase(self):
        testfilter = SimilarityFilter(threshold=0.7, unit='word', lowercase=True)
        expected = [([1], False), ([0], True), ([1], False),
                    ([1], False), ([1], False), ([0], True), ([0], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.bi_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_trilingual(self):
        testfilter = LongestCommonSubstringFilter(threshold=0.7, require_all=True)
        expected = [([1, 1, 1], False), ([1, 0, 0], False), ([0, 0, 0], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.tri_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_trilingual_any(self):
        testfilter = LongestCommonSubstringFilter(threshold=0.7, require_all=False)
        expected = [([1, 1, 1], False), ([1, 0, 0], True), ([0, 0, 0], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.tri_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)


class TestLangIDMethod(unittest.TestCase):

    pairs_inputs = [
        ("This sentence is in english", "Je suis une phrase en français"),
        ("me llamo bernardo", "je m'appelle Bernard")
    ]


class TestLangId(TestLangIDMethod):

    def test_accept(self):
        model = LanguageIDFilter(
            languages=['en', 'fr'], id_method='langid', thresholds=[0.8, 0.99])
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)

    def test_accept_with_set_languages(self):
        model = LanguageIDFilter(
            languages=['en', 'fr'], id_method='langid', thresholds=[0.8, 0.99],
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

    def test_accept(self):
        model = LanguageIDFilter(
            languages=['en', 'fr'], id_method='cld2', thresholds=[0.9, 0.9])
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)

    def test_accept_with_options(self):
        model = LanguageIDFilter(
            languages=['en', 'fr'], id_method='cld2', thresholds=[0.9, 0.9],
            cld2_options={'bestEffort': True})
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

    def test_accept(self):
        if self.testmodel is None:
            self.skipTest("Failed to download test resources")
        model = LanguageIDFilter(
            languages=['en', 'fr'], id_method='fasttext', thresholds=[0.8, 0.99],
            fasttext_model_path=self.testmodel)
        pair_scores = model.score(self.pairs_inputs)
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(pair_scores, pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)


class TestRepetitionFilter(unittest.TestCase):

    def test_get_repetition(self):
        segments_and_repetitions = [
            ('abc', [None]*3, [0, 0, 0]),
            ('aaa', [None]*3, [0, 0, 0]),
            ('abcabc', ['abc', None, None], [1, 0, 0]),
            ('abc abc', ['abc', None, None], [1, 0, 0]),
            ('abc abc abc', ['abc', 'abc', None], [2, 2, 0]),
            ('abc abc   abc', ['abc', 'abc', None], [2, 2, 0]),
            ('abc abcabc', ['abc', 'abc', None], [2, 2, 0]),
            ('abcabc   abc', ['abc', 'abc', None], [2, 2, 0]),
            ('abc abc   abcabc', ['abc']*3, [3, 3, 3]),
            ('aaaa aaaa aaaa', ['aaaa', 'aaaa', None], [2, 2, 0]),
            ('aaaa aaaa aaaa aaaa', ['aaaa']*3, [3, 3, 3]),
            ('aaaa aaaa bbbb aaaa aaaa bbbb aaaa aaaa bbbb aaaa aaaa bbbb',
             ['aaaa', 'aaaa aaaa bbbb', 'aaaa aaaa bbbb'], [1, 3, 3]),
            ('Ahora bien, el que quiera ser el primero entre ustedes deberá ser su servidor, diferentes plantas para '
             'ser un buen pescador y un buen pescador para ser un buen pescador y un buen pescador para ser un buen '
             'pescador y un buen pescador para ser un buen pescador',
             ['para ser un buen pescador y un buen pescador']*2 + [None], [2, 2, 0])
        ]
        for idx, threshold in enumerate(range(1, 4)):
            testfilter = RepetitionFilter(threshold, min_length=3, max_length=50)
            for segment, patterns, repetitions in segments_and_repetitions:
                logging.info("%s '%s'", threshold, segment)
                num, pat = testfilter.get_repetitions(segment)
                self.assertEqual(num, repetitions[idx])
                self.assertEqual(pat, patterns[idx])

    def test_bilingual(self):
        segments_and_decisions = [
            (('abcd', 'abcd'), True),
            (('abc abc', 'abcd'), True),
            (('abc abc', 'bcd bcd'), True),
            (('abc abc abc', 'bcd bcd'), False),
            (('abc', 'bcde bcde bcde bcde'), False),
        ]
        testfilter = RepetitionFilter(2, min_length=3, max_length=50)
        pairs, decisions = zip(*segments_and_decisions)
        scores = testfilter.score(pairs)
        for pair, score, expected in zip(pairs, scores, decisions):
            logger.info("%s %s %s", pair, score, expected)
            self.assertEqual(testfilter.accept(score), expected)
