import inspect
import logging
import os
import requests
import shutil
import tempfile
import unittest

import opustools

from opusfilter import FilterABC, ConfigurationError, filters, pipeline
from opusfilter.autogen import *
from opusfilter.opusfilter import OpusFilter


default_params = {
    'AlphabetRatioFilter': {},
    'CharacterScoreFilter': {'scripts': ['latin', 'latin']},
    'LanguageIDFilter': {'name': 'cld2', 'id_method': 'cld2', 'languages': ['en', 'de']},
    'LengthRatioFilter.char': {'name': 'char', 'unit': 'char'},
    'LengthRatioFilter.word': {'name': 'word', 'unit': 'word'},
    'NonZeroNumeralsFilter': {},
    'TerminalPunctuationFilter': {}
}

example_params = {
    'AlphabetRatioFilter': {'threshold': 1},
    'CharacterScoreFilter': {'scripts': ['latin', 'latin'], 'thresholds': [1, 1]},
    'LanguageIDFilter': {'name': 'cld2', 'id_method': 'cld2',
                         'languages': ['en', 'de'],
                         'thresholds': [1, 1]},
    'LengthRatioFilter.char': {'name': 'char', 'threshold': 1, 'unit': 'char'},
    'LengthRatioFilter.word': {'name': 'word', 'threshold': 1, 'unit': 'word'},
    'NonZeroNumeralsFilter': {'threshold': 1},
    'TerminalPunctuationFilter': {'threshold': 1}
}

default_rejects = {
    'TerminalPunctuationFilter': False,
    'AlphabetRatioFilter.0': False,
    'AlphabetRatioFilter.1': False,
    'CharacterScoreFilter.0': False,
    'CharacterScoreFilter.1': False,
    'LanguageIDFilter.0': False,
    'LanguageIDFilter.1': False,
    'LengthRatioFilter.char': False,
    'LengthRatioFilter.word': False,
    'NonZeroNumeralsFilter.0': False
}


class TestAutogen(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempdir = tempfile.mkdtemp()
        self.source = 'en'
        self.target = 'sv'
        self.src_out = os.path.join(self.tempdir, f'sents.{self.source}')
        self.tgt_out = os.path.join(self.tempdir, f'sents.{self.target}')
        opus_reader = opustools.OpusRead(
            directory='RF',
            source=self.source,
            target=self.target,
            release='v1',
            suppress_prompts=True,
            preprocess='raw',
            write_mode='moses',
            write=[self.src_out, self.tgt_out],
            leave_non_alignments_out=True,
            download_dir=self.tempdir)
        opus_reader.printPairs()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tempdir)

    def _test_filters(self, filters):
        self.assertTrue(filters)
        logging.info(filters)
        generator = ConfigurationGenerator(
            [self.src_out, self.tgt_out], workdir=os.path.join(self.tempdir, 'work'), langs=[self.source, self.target])
        generator.add_filter(filters)
        configuration = generator.get_config()
        of = OpusFilter(configuration)
        of.execute_steps(overwrite=True)
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir, 'work', f'filtered.{self.source}.gz')))
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir, 'work', f'filtered.{self.target}.gz')))

    def test_default_filters(self):
        filtergen = DefaultParameterFilters(
            langs=[self.source, self.target], scripts=['Latin', 'Latin'])
        filtergen.set_filter_thresholds()
        for spec in DefaultParameterFilters.DEFAULT_FILTERS:
            filter_name, _ = parse_filter_specs(spec)
            self.assertTrue(any(filter_name in f for f in filtergen.filters))
        self._test_filters(filtergen.filters)

    @unittest.expectedFailure
    def test_percentile_filters(self):
        filtergen = PercentileFilters(
            files=[self.src_out, self.tgt_out], langs=[self.source, self.target], scripts=['Latin', 'Latin'],
            excluded_percentile=0.05)
        filtergen.set_filter_thresholds()
        logging.info(filtergen.filters)
        self._test_filters(filtergen.filters)

    def test_threshold_finder(self):
        filtergen = ClusterFilters(
            files=[self.src_out, self.tgt_out], langs=[self.source, self.target], scripts=['Latin', 'Latin'],
            sample_size=180, inter_dir=self.tempdir, overwrite=True)
        filtergen.set_filter_thresholds()
        self._test_filters(filtergen.filters)


class TestThresholdFinder(unittest.TestCase):

    def test_set_default_parameters(self):
        tf = ClusterFilters([None, None], ['en', 'de'], ['latin', 'latin'], None, None, None)
        self.assertEqual(tf.filter_params, default_params)

    def test_reject_all_parameters(self):
        tf = ClusterFilters([None, None], ['en', 'de'], ['latin', 'latin'], None, None, None)
        tf._set_parameters([1 for i in range(10)], {k: True for k in default_rejects.keys()})
        self.assertEqual(tf.filter_params, {})

    def test_set_all_parameters(self):
        tf = ClusterFilters([None, None], ['en', 'de'], ['latin', 'latin'], None, None, None)
        tf._set_parameters([1 for i in range(10)], default_rejects)
        self.assertEqual(tf.filter_params, example_params)

    def test_set_parameters_reject_one_side(self):
        tf = ClusterFilters([None, None], ['en', 'de'], ['latin', 'latin'], None, None, None)
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


class TestGenericFilterAdjuster(unittest.TestCase):

    # These have arguments without defaults
    expected_failures = {'CharacterScoreFilter', 'CrossEntropyFilter', 'CrossEntropyDifferenceFilter',
                         'LMClassifierFilter', 'LanguageIDFilter', 'RegExpFilter', 'SentenceEmbeddingFilter'}

    def test_default_parameters(self):
        for filter_name, filter_cls in inspect.getmembers(filters, inspect.isclass):
            if not issubclass(filter_cls, FilterABC) or filter_cls == FilterABC:
                continue
            adjuster = GenericFilterAdjuster(filter_name)
            params = adjuster.default_parameters
            logging.info("%s %s", filter_name, params)
            if filter_name in self.expected_failures:
                with self.assertRaises((ConfigurationError, ModuleNotFoundError)):
                    obj = filter_cls(**params)
            else:
                try:
                    obj = filter_cls(**params)
                except ModuleNotFoundError:
                    logger.info("Skipping test for %s: Requred module not found", filter_name)

    @unittest.expectedFailure
    def test_adjusted_parameters(self):
        src_data = ['a'] * 11 + ['a bbbbb'] * 78 + ['a bbbbb cccc'] * 11
        tgt_data = [seg.upper() for seg in src_data]
        data = list(zip(src_data, tgt_data))
        for filter_name, filter_cls in inspect.getmembers(filters, inspect.isclass):
            if not issubclass(filter_cls, FilterABC) or filter_cls == FilterABC:
                continue
            if filter_name in self.expected_failures:
                continue
            adjuster = GenericFilterAdjuster(filter_name)
            if not adjuster.is_adjustable():
                continue
            params = adjuster.get_adjusted_parameters(data, excluded_percentile=0.1)
            logging.info("%s %s", filter_name, params)
            obj = filter_cls(**params)
            filtered = list(obj.filter(data))
            self.assertGreater(len(filtered), 0)
