"""Configuration generation tools"""

import copy
import inspect
import logging

from pandas import json_normalize
from tqdm import tqdm

from . import CLEAN_LOW, CLEAN_HIGH, CLEAN_BETWEEN, CLEAN_TRUE, CLEAN_FALSE
from . import OpusFilterError, ConfigurationError
from . import filters as filtermodule
from . import pipeline
from .classifier import lists_to_dicts
from .util import file_open


logger = logging.getLogger(__name__)
logger.setLevel('WARNING')


class FilterArgumentFailure(OpusFilterError):
    """Unusable default arguments for filter"""


class ConfigurationGenerator:
    """OpusFilter configuration generator"""

    def __init__(self, files, workdir='work', langs=None):
        self.files = files
        if langs and len(files) != len(langs):
            raise OpusFilterError("Number of files and languages must match")
        self.langs = langs
        self.workdir = workdir
        self.filters_to_add = []
        self.steps = []
        self.current_inputs = self.files

    def get_filenames(self, prefix, ext='gz'):
        """Return filenames with given prefix and extension"""
        if self.langs:
            return [f'{prefix}.{lang}.{ext}' for lang in self.langs]
        num = len(self.files)
        return [f'{prefix}.{i}.{ext}' for i in range(1, num + 1)]

    def add_remove_duplicates(self, prefix='dedup', update_input=True):
        """Add remove_duplicates to the configuration steps"""
        outputs = self.get_filenames(prefix)
        self.steps.append({
            'type': 'remove_duplicates',
            'parameters': {
                'inputs': self.current_inputs,
                'outputs': outputs
            }
        })
        if update_input:
            self.current_inputs = outputs
        return outputs

    def add_subset(self, size, seed, prefix='sample', update_input=True):
        """Add subset to the configuration steps"""
        outputs = self.get_filenames(prefix)
        self.steps.append({
            'type': 'subset',
            'parameters': {
                'inputs': self.current_inputs,
                'outputs': outputs,
                'size': size,
                'seed': seed
            }
        })
        if update_input:
            self.current_inputs = outputs
        return outputs

    def add_filter(self, filters, prefix='filtered', update_input=True):
        """Add filter to the configuration steps"""
        outputs = self.get_filenames(prefix)
        self.steps.append({
            'type': 'filter',
            'parameters': {
                'inputs': self.current_inputs,
                'outputs': outputs,
                'filters': filters
            }
        })
        if update_input:
            self.current_inputs = outputs
        return outputs

    def add_score(self, filters, prefix='scores', update_input=True):
        """Add score to the configuration steps"""
        output = f'{prefix}.jsonl.gz'
        self.steps.append({
            'type': 'score',
            'parameters': {
                'inputs': self.current_inputs,
                'output': output,
                'filters': filters
            }
        })
        if update_input:
            self.current_inputs = output
        return output

    def get_config(self):
        """Return current configuration"""
        config = {
            'common': {
                'output_directory': self.workdir
            },
            'steps': self.steps
        }
        return config


class DefaultParameterFilters:
    """Filter configuration with default parameters"""

    def __init__(self, filters=None):
        self.filters_to_add = filters if filters is not None else [
            'LengthFilter', 'LengthRatioFilter', 'LongWordFilter', 'HtmlTagFilter',
            'AverageWordLengthFilter', 'AlphabetRatioFilter',
            'TerminalPunctuationFilter', 'NonZeroNumeralsFilter',
            'LongestCommonSubstringFilter', 'SimilarityFilter', 'RepetitionFilter'
        ]

    def get_thresholds(self):
        """Get filter configuration with thresholds"""
        filters = []
        for filterclass in self.filters_to_add:
            try:
                filter_config = self.get_filter_parameters(filterclass)
            except FilterArgumentFailure as err:
                logger.error("Unusable default arguments for %s: %s", filterclass, err)
                continue
            filters.append(filter_config)
        return filters

    @staticmethod
    def get_filter_parameters(filterclass):
        """Return default parameters for filter of the given class"""
        adjuster = GenericFilterAdjuster(filterclass)
        filter_cls = getattr(filtermodule, filterclass)
        try:
            filter_cls(**adjuster.default_parameters)
        except ConfigurationError as err:
            raise FilterArgumentFailure(err) from err
        filter_config = {filterclass: adjuster.default_parameters}
        return filter_config


class PercentileFilters:
    """Configuration generator based on filter score percentiles"""

    def __init__(self, files, filters=None, excluded_percentile=0.001, sample_size=100000):
        self.files = files
        self.sample_size = sample_size
        self.excluded_percentile = excluded_percentile
        self.filters_to_add = filters if filters is not None else [
            'LengthFilter', 'LengthRatioFilter', 'LongWordFilter', 'HtmlTagFilter',
            'AverageWordLengthFilter', 'AlphabetRatioFilter',
            'TerminalPunctuationFilter', 'NonZeroNumeralsFilter',
            'LongestCommonSubstringFilter', 'SimilarityFilter', 'RepetitionFilter'
        ]

    def get_thresholds(self):
        """Get filter configuration with thresholds"""
        filters = []
        for filterclass in self.filters_to_add:
            try:
                filter_config = self.get_filter_parameters(filterclass)
            except FilterArgumentFailure as err:
                logger.error("Unusable default arguments for %s: %s", filterclass, err)
                continue
            filters.append(filter_config)
        return filters

    def get_filter_parameters(self, filterclass):
        """Return suitable parameters for filter of the given class"""
        adjuster = GenericFilterAdjuster(filterclass)
        filter_cls = getattr(filtermodule, filterclass)
        try:
            filter_cls(**adjuster.default_parameters)
        except ConfigurationError as err:
            raise FilterArgumentFailure(err) from err
        new_parameters = adjuster.get_adjusted_parameters(
            self.read_lines(), excluded_percentile=self.excluded_percentile)
        filter_config = {filterclass: new_parameters}
        return filter_config

    def read_lines(self):
        """Read segments without newlines"""
        infs = [file_open(infile, 'r') for infile in self.files]
        for pair in tqdm(zip(*infs)):
            yield [segment.rstrip() for segment in pair]


class GenericFilterAdjuster:
    """Class for guessing suitable parameters for a filter"""

    # Lists of possible filter threshold arguments
    SINGLE_THRESHOLD_ARGUMENTS = ['threshold']
    MULTI_THRESHOLD_ARGUMENTS = ['threshold', 'thresholds']
    MIN_MAX_ARGUMENTS = [('min_length', 'max_length')]
    ALL_THRESHOLD_ARGUMENTS = SINGLE_THRESHOLD_ARGUMENTS + MULTI_THRESHOLD_ARGUMENTS + MIN_MAX_ARGUMENTS

    def __init__(self, filterclass):
        if isinstance(filterclass, str):
            self.filter_name = filterclass
            self.filter_cls = getattr(filtermodule, self.filter_name)
        else:
            self.filter_name = filterclass.__name__
            self.filter_cls = filterclass
        self.default_parameters = self.get_default_parameters()

    def get_default_parameters(self):
        """Get default parameters for the filter

        Uses the signature of the class. Arguments without default
        values are ignored and will cause a failure.

        """
        default_parameters = {}
        sig = inspect.signature(self.filter_cls)
        logger.info("signature: %s%s", self.filter_name, sig)
        for key, parameter in sig.parameters.items():
            if parameter.default == inspect.Signature.empty:
                if key != 'kwargs':
                    logger.warning("Ignoring argument without default: %s", key)
                continue
            default_parameters[key] = parameter.default
        return default_parameters

    @staticmethod
    def _locate_arguments(candidates, arguments):
        """Return first matching candidate key / tuple of keys from arguments

        For tuples in candidates, all keys in the tuple must be found
        from arguments.

        """
        for candidate in candidates:
            if isinstance(candidate, str):
                if candidate in arguments:
                    return candidate
            else:
                if all(key in arguments for key in candidate):
                    return candidate
        return None

    def is_adjustable(self):
        """Return whether the current filter has parameters to adjust"""
        if self.filter_cls.score_direction in {CLEAN_TRUE, CLEAN_FALSE}:
            # Nothing to estimate for boolean
            return False
        if self._locate_arguments(self.ALL_THRESHOLD_ARGUMENTS, self.default_parameters):
            # Known threshold parameters to adjust
            return True
        return False

    def get_adjusted_parameters(self, data, excluded_percentile=0.01):
        """Estimate parameters for the filter using data

        Assumes that excluded_percentile amount of the data should be
        excluded by the filter. Depending on filter's score_direction,
        they might be the lowest values, the highest values, or both
        the lowest and highest values.

        """
        # TODO: It should be checked that the selected threshold does
        # not remove too much data. E.g. if the clean values are high,
        # excluded_percentile is 1%, and the highest 99.1% of the
        # values are 1, the selected threshold 1 might remove all data
        # if the condition for accepting the value is being greater
        # than the threshold.
        parameters = copy.deepcopy(self.default_parameters)
        if not self.is_adjustable():
            return parameters
        score_dir = self.filter_cls.score_direction
        logger.info("score type for %s: %s", self.filter_name, score_dir)
        filter_config = {self.filter_name: self.default_parameters}
        filter_pipe = pipeline.FilterPipeline.from_config([filter_config])
        df = self.get_score_df(filter_pipe, data)
        if score_dir == CLEAN_LOW:
            percentiles = [1 - excluded_percentile]
            pct_keys = [f'{100*(1-excluded_percentile):g}%']
        elif score_dir == CLEAN_HIGH:
            percentiles = [excluded_percentile]
            pct_keys = [f'{100*excluded_percentile:g}%']
        elif score_dir == CLEAN_BETWEEN:
            half_pct = excluded_percentile / 2
            percentiles = [half_pct, 1 - half_pct]
            pct_keys = [f'{100*half_pct:g}%', f'{100*(1-half_pct):g}%']
        else:
            raise ValueError(f"Unknown score type '{score_dir}'")
        score_dim = len(df.columns)
        if score_dir in {CLEAN_LOW, CLEAN_HIGH}:
            # Clean is below or above threshold
            if score_dim > 1:
                # Multiple thresholds allowed (or needed)
                threshold_key = self._locate_arguments(self.MULTI_THRESHOLD_ARGUMENTS, parameters)
            else:
                # Single threshold
                threshold_key = self._locate_arguments(self.SINGLE_THRESHOLD_ARGUMENTS, parameters)
            if not threshold_key:
                logger.warning("Cannot find threshold parameter from %s", list(parameters))
                return parameters
            values = []
            for column in df.columns:
                stats = df[column].describe(percentiles=percentiles)
                logger.info(stats)
                values.append(stats.loc[pct_keys[0]].item())
            if score_dim == 1:
                values = values[0]
            logger.info("Selected value %s for %s", values, threshold_key)
            parameters[threshold_key] = values
        elif score_dir == CLEAN_BETWEEN:
            # Clean is between minimum and maximum
            threshold_key = self._locate_arguments(self.MIN_MAX_ARGUMENTS, parameters)
            if not threshold_key:
                logger.warning("Cannot find threshold parameter from %s", list(parameters))
                return parameters
            min_values, max_values = [], []
            for column in df.columns:
                stats = df[column].describe(percentiles=percentiles)
                logger.info(stats)
                min_values.append(stats.loc[pct_keys[0]].item())
                max_values.append(stats.loc[pct_keys[1]].item())
            if score_dim == 1:
                min_values = min_values[0]
                max_values = max_values[0]
            logger.info("Selected values %s - %s for %s", min_values, max_values, threshold_key)
            parameters[threshold_key[0]] = min_values
            parameters[threshold_key[1]] = max_values
        else:
            logger.warning("Threshold adjusting not supported")
        return parameters

    @staticmethod
    def get_score_df(filter_pipe, data):
        """Return dataframe containing filter scores from the data"""
        return json_normalize([lists_to_dicts(scores_obj) for scores_obj in filter_pipe.score(data)])
