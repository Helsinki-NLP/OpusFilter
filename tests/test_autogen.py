import inspect
import logging
import os
import requests
import shutil
import tempfile
import unittest

from opusfilter import FilterABC, ConfigurationError, filters, pipeline
from opusfilter.autogen import *


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
                with self.assertRaises(ConfigurationError):
                    obj = filter_cls(**params)
            else:
                obj = filter_cls(**params)

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
