import copy
import unittest

from opusfilter.pipeline import FilterPipeline


class TestFilterPipeline(unittest.TestCase):

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
                {'LanguageIDFilter': {'languages': ['en', 'sv'],
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
        self.assertEqual(
            scores[0],
            {'LengthFilter': [5, 9],
             'LengthRatioFilter': 1.8,
             'LongWordFilter': [12, 11],
             'AverageWordLengthFilter': [6, 19 / 3],
             'HtmlTagFilter': [False, False],
             'CharacterScoreFilter': [1.0, 1.0],
             'LanguageIDFilter': [1.0, 1.0],
             'TerminalPunctuationFilter': -0.0,
             'NonZeroNumeralsFilter': [1.0]})
        self.assertEqual(
            scores[1],
            {'LengthFilter': [1, 1],
             'LengthRatioFilter': 1.0,
             'LongWordFilter': [6, 10],
             'AverageWordLengthFilter': [6, 10],
             'HtmlTagFilter': [False, False],
             'CharacterScoreFilter': [1.0, 1.0],
             'LanguageIDFilter': [0.17, 0.0],
             'TerminalPunctuationFilter': -2.1972245773362196,
             'NonZeroNumeralsFilter': [0.8888888888888888]})
        self.assertEqual(
            scores[2],
            {'LengthFilter': [0, 0],
             'LengthRatioFilter': 0,
             'LongWordFilter': [0, 0],
             'AverageWordLengthFilter': [0, 0],
             'HtmlTagFilter': [False, False],
             'CharacterScoreFilter': [1.0, 1.0],
             'LanguageIDFilter': [1.0, 1.0],
             'TerminalPunctuationFilter': -0.0,
             'NonZeroNumeralsFilter': [1.0]})

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


class TestFilterPipelineScoreNames(unittest.TestCase):

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
        self.assertEqual(
            scores[0],
            {'LengthFilter': {'1': [5, 9], '2': [34, 65]}})
        self.assertEqual(
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
        self.assertEqual(
            scores[0],
            {'LengthFilter': {'words': [5, 9], 'chars': [34, 65]}})
        self.assertEqual(
            scores[1],
            {'LengthFilter': {'words': [1, 1], 'chars': [6, 10]}})
