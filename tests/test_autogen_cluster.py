import unittest
import pprint
import copy

from opusfilter.autogen_cluster import FilterThresholdFinder

default_params = {'AlphabetRatioFilter': {},
     'CharacterScoreFilter': {'scripts': ['latin', 'latin']},
     'LanguageIDFilter': {'name': 'cld2', 'id_method': 'cld2', 'languages': ['en', 'de']},
     'LengthRatioFilter.char': {'name': 'char', 'unit': 'char'},
     'LengthRatioFilter.word': {'name': 'word', 'unit': 'word'},
     'NonZeroNumeralsFilter': {},
     'TerminalPunctuationFilter': {}}

example_params = {'AlphabetRatioFilter': {'threshold': 1},
     'CharacterScoreFilter': {'scripts': ['latin', 'latin'], 'thresholds': [1, 1]},
     'LanguageIDFilter': {'name': 'cld2', 'id_method': 'cld2',
                          'languages': ['en', 'de'],
                          'thresholds': [1, 1]},
     'LengthRatioFilter.char': {'name': 'char', 'threshold': 1, 'unit': 'char'},
     'LengthRatioFilter.word': {'name': 'word', 'threshold': 1, 'unit': 'word'},
     'NonZeroNumeralsFilter': {'threshold': 1},
     'TerminalPunctuationFilter': {'threshold': 1}}

default_rejects = {'TerminalPunctuationFilter': False,
        'AlphabetRatioFilter.0': False,
        'AlphabetRatioFilter.1': False,
        'CharacterScoreFilter.0': False,
        'CharacterScoreFilter.1': False,
        'LanguageIDFilter.0': False,
        'LanguageIDFilter.1': False,
        'LengthRatioFilter.char': False,
        'LengthRatioFilter.word': False,
        'NonZeroNumeralsFilter.0': False}


class TestThresholdFinder(unittest.TestCase):

    def test_set_default_parameters(self):
        tf = FilterThresholdFinder([None, None], ['en', 'de'], ['latin', 'latin'], None, None, None)
        self.assertEqual(tf.filter_params, default_params)

    def test_reject_all_parameters(self):
        tf = FilterThresholdFinder([None, None], ['en', 'de'], ['latin', 'latin'], None, None, None)
        tf._set_parameters([1 for i in range(10)], {k: True for k in default_rejects.keys()})
        self.assertEqual(tf.filter_params, {})

    def test_set_all_parameters(self):
        tf = FilterThresholdFinder([None, None], ['en', 'de'], ['latin', 'latin'], None, None, None)
        tf._set_parameters([1 for i in range(10)], default_rejects)
        self.assertEqual(tf.filter_params, example_params)

    def test_set_parameters_reject_one_side(self):
        tf = FilterThresholdFinder([None, None], ['en', 'de'], ['latin', 'latin'], None, None, None)
        rejects = copy.deepcopy(default_rejects)
        rejects['LanguageIDFilter.0'] = True
        params = copy.deepcopy(example_params)
        params['LanguageIDFilter']['thresholds'][0] = -1
        tf._set_parameters([1 for i in range(10)], rejects)
        self.assertEqual(tf.filter_params, params)

        tf.filter_params = copy.deepcopy(default_params)
        rejects = copy.deepcopy(default_rejects)
        rejects['LanguageIDFilter.1'] = True
        params = copy.deepcopy(example_params)
        params['LanguageIDFilter']['thresholds'][1] = -1
        tf._set_parameters([1 for i in range(10)], rejects)
        self.assertEqual(tf.filter_params, params)
        
        tf.filter_params = copy.deepcopy(default_params)
        rejects = copy.deepcopy(default_rejects)
        rejects['CharacterScoreFilter.0'] = True
        params = copy.deepcopy(example_params)
        params['CharacterScoreFilter']['thresholds'][0] = -1
        tf._set_parameters([1 for i in range(10)], rejects)
        self.assertEqual(tf.filter_params, params)
        
        tf.filter_params = copy.deepcopy(default_params)
        rejects = copy.deepcopy(default_rejects)
        rejects['CharacterScoreFilter.1'] = True
        params = copy.deepcopy(example_params)
        params['CharacterScoreFilter']['thresholds'][1] = -1
        tf._set_parameters([1 for i in range(10)], rejects)
        self.assertEqual(tf.filter_params, params)
