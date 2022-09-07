"""Processor for filter configurations"""

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


class FilterArgumentFailure(OpusFilterError):
    """Unusable default arguments for filter"""


class GenericFilterAdjuster:
    """Class for guessing suitable parameters for a filter"""

    # Lists of possible filter threshold arguments
    SINGLE_THRESHOLD_ARGUMENTS = ['threshold']
    MULTI_THRESHOLD_ARGUMENTS = ['threshold', 'thresholds']
    MIN_MAX_ARGUMENTS = [('min_length', 'max_length')]

    def __init__(self, filterclass):
        self.filterclass = filterclass
        self.default_parameters = self.get_default_parameters()

    def get_default_parameters(self):
        """Get default parameters for the filter

        Uses the signature of the class. Arguments without default
        values are ignored and will cause a failure.

        """
        filter_cls = getattr(filtermodule, self.filterclass)
        default_parameters = {}
        sig = inspect.signature(filter_cls)
        logger.info("signature: %s%s", self.filterclass, sig)
        for key, parameter in sig.parameters.items():
            if parameter.default == inspect._empty:
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
        filter_cls = getattr(filtermodule, self.filterclass)
        score_dir = filter_cls.score_direction
        logger.info("score type for %s: %s", self.filterclass, score_dir)
        if score_dir in {CLEAN_TRUE, CLEAN_FALSE}:
            # Nothing to estimate
            return parameters
        filter_config = {self.filterclass: self.default_parameters}
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
            raise ValueError("Unknown score type '%s'", score_dir)
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

    def get_score_df(self, filter_pipe, data):
        """Return dataframe containing filter scores from the data"""
        return json_normalize([lists_to_dicts(scores_obj) for scores_obj in filter_pipe.score(data)])


class ConfigurationGenerator:
    """OpusFilter configuration generator"""

    # TODO:
    # - Check likely languages from the data for LanguageIDFilter
    # - Check likely character sets from the data for CharacterScoreFilter
    # - Add options to override the automatically generated filter parameters
    # - Separate input files to filter from those used for setting the thresholds
    # - Specify input data from OPUS instead of providing files
    # - Optional train/devel/test data splits
    # - Duplicate filtering
    # - ...

    def __init__(self, files, workdir='work', excluded_percentile=0.001):
        self.files = files
        self.workdir = workdir
        self.excluded_percentile = excluded_percentile
        self.filters_to_add = [
            'LengthFilter', 'LengthRatioFilter', 'LongWordFilter', 'HtmlTagFilter',
            'AverageWordLengthFilter', 'AlphabetRatioFilter',
            'TerminalPunctuationFilter', 'NonZeroNumeralsFilter',
            'LongestCommonSubstringFilter', 'SimilarityFilter', 'RepetitionFilter'
        ]
        self.filters = []
        for filterclass in self.filters_to_add:
            try:
                filter_config = self.get_filter_parameters(filterclass)
            except FilterArgumentFailure as err:
                logger.error("Unusable default arguments for %s: %s", filterclass, err)
                continue
            self.filters.append(filter_config)

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

    def get_filenames(self, prefix, ext='gz'):
        """Return filenames with given prefix and extension"""
        # TODO: Try to find language codes or other changing parts
        # from the input files and use them instead of indices
        num = len(self.files)
        return [f'{prefix}.{i}.{ext}' for i in range(1, num + 1)]

    def add_filter(self, steps):
        """Add filter to the configuration steps"""
        steps.append({
            'type': 'filter',
            'parameters': {
                'inputs': self.files,
                'outputs': self.get_filenames('filtered'),
                'filters': self.filters
            }
        })

    def get_config(self):
        """Return current configuration"""
        steps = []
        self.add_filter(steps)
        config = {
            'common': {
                'output_directory': self.workdir
            },
            'steps': steps
        }
        return config
