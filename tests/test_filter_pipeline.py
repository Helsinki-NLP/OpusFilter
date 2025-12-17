import copy
import unittest

from numpy.testing import assert_almost_equal

from opusfilter.pipeline import FilterPipeline


class TestFilterPipelineBase(unittest.TestCase):

    def assert_scores_equal(self, sdict1, sdict2):
        self.assertEqual(set(sdict1), set(sdict2))  # same keys
        for key, val1 in sdict1.items():
            val2 = sdict2[key]
            if isinstance(val1, list):
                self.assertEqual(len(val1), len(val2), msg=f"Scores do not match for {key}: {val1} {val2}")
                for item1, item2 in zip(val1, val2):
                    self.assertAlmostEqual(item1, item2, msg=f"Scores do not match for {key}: {val1} {val2}")
            else:
                self.assertAlmostEqual(val1, val2, msg=f"Scores do not match for {key}: {val1} {val2}")

    def assert_scores_and_decisions_equal(self, sdict1, sdict2):
        self.assertEqual(set(sdict1), set(sdict2))  # same keys
        for key, res1 in sdict1.items():
            res2 = sdict2[key]
            acc1, acc2 = res1['accept'], res2['accept']
            val1, val2 = res1['scores'], res2['scores']
            self.assertEqual(acc1, acc2, msg=f"Decisions do not match for {key}: {acc1} {acc2}")
            if isinstance(val1, list):
                self.assertEqual(len(val1), len(val2), msg=f"Scores do not match for {key}: {val1} {val2}")
                for item1, item2 in zip(val1, val2):
                    self.assertAlmostEqual(item1, item2, msg=f"Scores do not match for {key}: {val1} {val2}")
            else:
                self.assertAlmostEqual(val1, val2, msg=f"Scores do not match for {key}: {val1} {val2}")


class TestFilterPipeline(TestFilterPipelineBase):

    @classmethod
    def setUpClass(self):
        self.config = [
                {'LengthFilter': {'min_length': 1, 'max_length': 100,
                    'unit': 'word'}},
                {'LengthRatioFilter': {'threshold': 3, 'unit': 'word'}},
                {'LongWordFilter': {'threshold': 40}},
                {'AverageWordLengthFilter': {'min_length': 1, 'max_length': 20}},
                {'HtmlTagFilter': {}},
                {'CharacterScoreFilter': {'scripts': ['Latin', 'Latin'],
                                          'thresholds': [1, 1]}},
                {'LangidFilter': {'languages': ['en', 'sv'],
                                  'thresholds': [0, 0]}},
                {'TerminalPunctuationFilter': {'threshold': -2}},
                {'NonZeroNumeralsFilter': {'threshold': 0.5}}
           ]

    def test_from_config(self):
        fp = FilterPipeline.from_config(self.config)
        self.assertEqual(len(fp.filters), 9)

    def test_set_chunksize(self):
        fp = FilterPipeline.from_config(self.config)
        fp.chunksize = 100

    def test_set_chunksize_value_error(self):
        fp = FilterPipeline.from_config(self.config)
        with self.assertRaises(ValueError):
            fp.chunksize = None
        with self.assertRaises(ValueError):
            fp.chunksize = 0

    def test_score(self):
        fp = FilterPipeline.from_config(self.config)
        pairs = [('That safeguards our independence .',
                  ('Kränkningar av svenskt territorium kommer aldrig att '
                   'accepteras .')),
                 ('1245..', '12345.....'),
                 ('', '')]
        scores = list(fp.score(pairs))
        self.assert_scores_equal(
            scores[0],
            {'LengthFilter': [5, 9],
             'LengthRatioFilter': 1.8,
             'LongWordFilter': [12, 11],
             'AverageWordLengthFilter': [6, 19 / 3],
             'HtmlTagFilter': [False, False],
             'CharacterScoreFilter': [1.0, 1.0],
             'LangidFilter': [1.0, 1.0],
             'TerminalPunctuationFilter': -0.0,
             'NonZeroNumeralsFilter': [1.0]})
        self.assert_scores_equal(
            scores[1],
            {'LengthFilter': [1, 1],
             'LengthRatioFilter': 1.0,
             'LongWordFilter': [6, 10],
             'AverageWordLengthFilter': [6, 10],
             'HtmlTagFilter': [False, False],
             'CharacterScoreFilter': [1.0, 1.0],
             'LangidFilter': [0.17, 0.0],
             'TerminalPunctuationFilter': -2.1972245773362196,
             'NonZeroNumeralsFilter': [0.8888888888888888]})
        self.assert_scores_equal(
            scores[2],
            {'LengthFilter': [0, 0],
             'LengthRatioFilter': 0,
             'LongWordFilter': [0, 0],
             'AverageWordLengthFilter': [0, 0],
             'HtmlTagFilter': [False, False],
             'CharacterScoreFilter': [1.0, 1.0],
             'LangidFilter': [1.0, 1.0],
             'TerminalPunctuationFilter': -0.0,
             'NonZeroNumeralsFilter': [1.0]})

    def test_score_with_decision(self):
        fp = FilterPipeline.from_config(self.config)
        pairs = [('That safeguards our independence .',
                  ('Kränkningar av svenskt territorium kommer aldrig att '
                   'accepteras .')),
                 ('1245..', '12345.....'),
                 ('', '')]
        scores = list(fp.score(pairs, with_decision=True))
        self.assert_scores_and_decisions_equal(
            scores[0],
            {'LengthFilter': {'scores': [5, 9], 'accept': True},
             'LengthRatioFilter': {'scores': 1.8, 'accept': True},
             'LongWordFilter': {'scores': [12, 11], 'accept': True},
             'AverageWordLengthFilter': {'scores': [6, 19 / 3], 'accept': True},
             'HtmlTagFilter': {'scores': [False, False], 'accept': True},
             'CharacterScoreFilter': {'scores': [1.0, 1.0], 'accept': True},
             'LangidFilter': {'scores': [1.0, 1.0], 'accept': True},
             'TerminalPunctuationFilter': {'scores': -0.0, 'accept': True},
             'NonZeroNumeralsFilter': {'scores': [1.0], 'accept': True}})
        self.assert_scores_and_decisions_equal(
            scores[1],
            {'LengthFilter': {'scores': [1, 1], 'accept': True},
             'LengthRatioFilter': {'scores': 1.0, 'accept': True},
             'LongWordFilter': {'scores': [6, 10], 'accept': True},
             'AverageWordLengthFilter': {'scores': [6, 10], 'accept': True},
             'HtmlTagFilter': {'scores': [False, False], 'accept': True},
             'CharacterScoreFilter': {'scores': [1.0, 1.0], 'accept': True},
             'LangidFilter': {'scores': [0.17, 0.0], 'accept': False},
             'TerminalPunctuationFilter': {'scores': -2.1972245773362196, 'accept': False},
             'NonZeroNumeralsFilter': {'scores': [0.8888888888888888], 'accept': True}})
        self.assert_scores_and_decisions_equal(
            scores[2],
            {'LengthFilter': {'scores': [0, 0], 'accept': False},
             'LengthRatioFilter': {'scores': 0, 'accept': True},
             'LongWordFilter': {'scores': [0, 0], 'accept': True},
             'AverageWordLengthFilter': {'scores': [0, 0], 'accept': False},
             'HtmlTagFilter': {'scores': [False, False], 'accept': True},
             'CharacterScoreFilter': {'scores': [1.0, 1.0], 'accept': True},
             'LangidFilter': {'scores': [1.0, 1.0], 'accept': True},
             'TerminalPunctuationFilter': {'scores': -0.0, 'accept': True},
             'NonZeroNumeralsFilter': {'scores': [1.0], 'accept': True}})

    def test_filter(self):
        fp = FilterPipeline.from_config(self.config)
        pairs = [('test', ''),
                (' '.join(['w' for i in range(101)]), 'test'),
                (''.join(['c' for i in range(41)]), 'test'),
                ('<s>test', 'test'),
                ('test', 'Φtest'),
                ('Tämä lause on kirjoitettu suomeksi.',
                    'This sentence is written in English.'),
                ('test', 'test...............'),
                ('1', '99999999999'),
                ('This sentence is written in English.',
                    'Denna mening är skriven på svenska.')]
        filtered = list(fp.filter(pairs))
        self.assertEqual(filtered, [('This sentence is written in English.',
                    'Denna mening är skriven på svenska.')])
        rev_pairs = [p for p in reversed(pairs)]
        filtered = list(fp.filter(rev_pairs))
        self.assertEqual(filtered, [('This sentence is written in English.',
                    'Denna mening är skriven på svenska.')])

    def test_filterfalse(self):
        fp = FilterPipeline.from_config(self.config)
        pairs = [('test', ''),
                (' '.join(['w' for i in range(101)]), 'test'),
                (''.join(['c' for i in range(41)]), 'test'),
                ('<s>test', 'test'),
                ('test', 'Φtest'),
                ('Tämä lause on kirjoitettu suomeksi.',
                    'This sentence is written in English.'),
                ('test', 'test...............'),
                ('1', '99999999999'),
                ('This sentence is written in English.',
                    'Denna mening är skriven på svenska.')]
        filtered = list(fp.filterfalse(pairs))
        self.assertEqual(filtered, pairs[:-1])

    def test_filter_empty(self):
        fp = FilterPipeline.from_config(self.config)
        pairs = [('', ''), ('this is English', 'det är Svenska'), ('', '')]
        filtered = list(fp.filter(pairs))
        self.assertEqual(filtered, [('this is English', 'det är Svenska')])
        # set LengthFilter to pass empty lines
        config2 = copy.deepcopy(self.config)
        config2[0]['LengthFilter']['pass_empty'] = True
        config2[3]['AverageWordLengthFilter']['pass_empty'] = True
        fp = FilterPipeline.from_config(config2)
        filtered = list(fp.filter(pairs))
        self.assertEqual(
            filtered, [('', ''), ('this is English', 'det är Svenska'), ('', '')])


class TestFilterPipelineScoreNames(TestFilterPipelineBase):

    def test_without_names(self):
        config = [
            {'LengthFilter': {'min_length': 1, 'max_length': 100,
                              'unit': 'word'}},
            {'LengthFilter': {'min_length': 8, 'max_length': 1000,
                              'unit': 'char'}},
        ]
        fp = FilterPipeline.from_config(config)
        self.assertEqual(len(fp.filters), 2)
        self.assertSequenceEqual(
            fp.get_score_tuples(),
            [('LengthFilter', '1'), ('LengthFilter', '2')])
        pairs = [('That safeguards our independence .',
                  ('Kränkningar av svenskt territorium kommer aldrig att '
                   'accepteras .')),
                 ('1245..',
                  '12345.....')]
        scores = list(fp.score(pairs))
        self.assert_scores_equal(
            scores[0],
            {'LengthFilter': {'1': [5, 9], '2': [34, 65]}})
        self.assert_scores_equal(
            scores[1],
            {'LengthFilter': {'1': [1, 1], '2': [6, 10]}})

    def test_with_names(self):
        config = [
            {'LengthFilter': {'min_length': 1, 'max_length': 100,
                              'unit': 'word', 'name': 'words'}},
            {'LengthFilter': {'min_length': 8, 'max_length': 1000,
                              'unit': 'char', 'name': 'chars'}},
        ]
        fp = FilterPipeline.from_config(config)
        self.assertEqual(len(fp.filters), 2)
        self.assertSequenceEqual(
            fp.get_score_tuples(),
            [('LengthFilter', 'words'), ('LengthFilter', 'chars')])
        pairs = [('That safeguards our independence .',
                  ('Kränkningar av svenskt territorium kommer aldrig att '
                   'accepteras .')),
                 ('1245..',
                  '12345.....')]
        scores = list(fp.score(pairs))
        self.assert_scores_equal(
            scores[0],
            {'LengthFilter': {'words': [5, 9], 'chars': [34, 65]}})
        self.assert_scores_equal(
            scores[1],
            {'LengthFilter': {'words': [1, 1], 'chars': [6, 10]}})
