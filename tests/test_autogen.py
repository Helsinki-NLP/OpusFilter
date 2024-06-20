import copy
import inspect
import logging
import os
import shutil
import tempfile
import unittest

import pandas as pd
from pandas import json_normalize

import opustools

from opusfilter import FilterABC, ConfigurationError, filters
from opusfilter.autogen import ConfigurationGenerator, DefaultParameterFilters, \
    parse_filter_specs, PercentileFilters, PercentileAdjuster, ClusterFilters
from opusfilter.opusfilter import OpusFilter
from opusfilter.pipeline import FilterPipeline
from opusfilter.util import lists_to_dicts


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

    col_names = [
        'AlphabetRatioFilter.0',
        'AlphabetRatioFilter.1',
        'LengthRatioFilter.char',
        'LengthRatioFilter.word',
        'NonZeroNumeralsFilter.0',
        'CharacterScoreFilter.0',
        'CharacterScoreFilter.1',
        'LanguageIDFilter.0',
        'LanguageIDFilter.1',
        'TerminalPunctuationFilter'
    ]

    example_params = [
        {'AlphabetRatioFilter': {'threshold': [1, 1]}},
        {'LengthRatioFilter': {'name': 'char', 'threshold': 1, 'unit': 'char'}},
        {'LengthRatioFilter': {'name': 'word', 'threshold': 1, 'unit': 'word'}},
        {'NonZeroNumeralsFilter': {'threshold': 1}},
        {'CharacterScoreFilter': {'scripts': ['latin', 'latin'], 'thresholds': [1, 1]}},
        {'LanguageIDFilter': {'id_method': 'lingua', 'languages': ['en', 'de'], 'thresholds': [1, 1]}},
        {'TerminalPunctuationFilter': {'threshold': 1}}
    ]

    def _make_df(self, names, thresholds, rejects):
        return pd.DataFrame.from_dict(
            {'name': names, 'threshold': thresholds, 'reject': rejects})

    def test_set_default_parameters(self):
        tf = ClusterFilters([None, None], langs=['en', 'de'], scripts=['latin', 'latin'])
        self.assertEqual(tf.filters, [])

    def test_reject_all_parameters(self):
        tf = ClusterFilters([None, None], langs=['en', 'de'], scripts=['latin', 'latin'])
        tf._set_parameters(self._make_df(self.col_names, [1] * len(self.col_names), [True] * len(self.col_names)))
        self.assertEqual(tf.filters, [])

    def test_set_all_parameters(self):
        tf = ClusterFilters([None, None], langs=['en', 'de'], scripts=['latin', 'latin'])
        tf._set_parameters(self._make_df(self.col_names, [1] * len(self.col_names), [False] * len(self.col_names)))
        self.assertSequenceEqual(tf.filters, self.example_params)

    def test_set_parameters_reject_one_side(self):
        tf = ClusterFilters([None, None], langs=['en', 'de'], scripts=['latin', 'latin'])
        default_rejects = [False] * len(self.col_names)
        rejects = copy.deepcopy(default_rejects)
        rejects[7] = True  # 'LanguageIDFilter.0'
        params = copy.deepcopy(self.example_params)
        params[5]['LanguageIDFilter']['thresholds'][0] = filters.LanguageIDFilter.accept_threshold
        tf._set_parameters(self._make_df(self.col_names, [1] * len(self.col_names), rejects))
        self.assertEqual(tf.filters, params)

        rejects = copy.deepcopy(default_rejects)
        rejects[8] = True  # LanguageIDFilter.1
        params = copy.deepcopy(self.example_params)
        params[5]['LanguageIDFilter']['thresholds'][1] = filters.LanguageIDFilter.accept_threshold
        tf._set_parameters(self._make_df(self.col_names, [1] * len(self.col_names), rejects))
        self.assertEqual(tf.filters, params)

        rejects = copy.deepcopy(default_rejects)
        rejects[5] = True  # 'CharacterScoreFilter.0'
        params = copy.deepcopy(self.example_params)
        params[4]['CharacterScoreFilter']['thresholds'][0] = filters.CharacterScoreFilter.accept_threshold
        tf._set_parameters(self._make_df(self.col_names, [1] * len(self.col_names), rejects))
        self.assertEqual(tf.filters, params)

        rejects = copy.deepcopy(default_rejects)
        rejects[6] = True  # 'CharacterScoreFilter.1'
        params = copy.deepcopy(self.example_params)
        params[4]['CharacterScoreFilter']['thresholds'][1] = filters.CharacterScoreFilter.accept_threshold
        tf._set_parameters(self._make_df(self.col_names, [1] * len(self.col_names), rejects))
        self.assertEqual(tf.filters, params)


class TestPercentileAdjuster(unittest.TestCase):

    # These have arguments without defaults
    expected_failures = {'CharacterScoreFilter', 'CrossEntropyFilter', 'CrossEntropyDifferenceFilter',
                         'LMClassifierFilter', 'LanguageIDFilter', 'RegExpFilter', 'SentenceEmbeddingFilter'}

    def test_default_parameters(self):
        for filter_name, filter_cls in inspect.getmembers(filters, inspect.isclass):
            if not issubclass(filter_cls, FilterABC) or filter_cls == FilterABC:
                continue
            adjuster = PercentileAdjuster(filter_name)
            params = adjuster.initial_parameters
            logging.info("%s %s", filter_name, params)
            if filter_name in self.expected_failures:
                with self.assertRaises((ConfigurationError, ModuleNotFoundError)):
                    obj = filter_cls(**params)
            else:
                try:
                    obj = filter_cls(**params)
                except ModuleNotFoundError:
                    logging.info("Skipping test for %s: Requred module not found", filter_name)

    def _get_score_df(self, filter_cls, data):
        pipeline = FilterPipeline([filter_cls()])
        df_data = [lists_to_dicts(score) for score in pipeline.score(data)]
        return pd.DataFrame(json_normalize(df_data))

    def test_adjusted_parameters(self):
        src_data = ['a'] * 11 + ['a bbbbb'] * 78 + ['a bbbbb cccc'] * 11
        tgt_data = [seg.upper() for seg in src_data]
        data = list(zip(src_data, tgt_data))
        self._test_adjusted_parameters(data)
        src_data += ['a bbbbb']
        tgt_data += ['A']
        data = list(zip(src_data, tgt_data))
        self._test_adjusted_parameters(data)

    def _test_adjusted_parameters(self, data):
        for filter_name, filter_cls in inspect.getmembers(filters, inspect.isclass):
            if not issubclass(filter_cls, FilterABC) or filter_cls == FilterABC:
                continue
            if filter_name in self.expected_failures:
                continue
            adjuster = PercentileAdjuster(filter_name)
            if not adjuster.is_adjustable():
                continue
            df = self._get_score_df(filter_cls, data)
            params = adjuster.get_adjusted_parameters(df, excluded_percentile=0.1)
            logging.info("%s %s", filter_name, params)
            obj = filter_cls(**params)
            filtered = list(obj.filter(data))
            self.assertGreater(len(filtered), 0)
