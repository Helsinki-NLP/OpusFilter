"""Configuration generation tools"""

import abc
import copy
import inspect
import logging
import os
import pathlib
import shutil
import tempfile

import ruamel.yaml

from . import CLEAN_LOW, CLEAN_HIGH, CLEAN_BETWEEN, CLEAN_TRUE, CLEAN_FALSE
from . import OpusFilterError, ConfigurationError
from . import filters as filtermodule
from .autogen_cluster import ScoreClusters
from .classifier import load_dataframe
from .opusfilter import OpusFilter


logger = logging.getLogger(__name__)
logger.setLevel('WARNING')
yaml = ruamel.yaml.YAML()


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


def get_score_file(input_files, filters, outputdir, sample_size, overwrite=False, max_length=150):
    """Calculate filter scores and return score file

    Remove duplicates and empty lines, take a sample of size n,
    produce filter scores file, and return its path.

    """
    config_gen = ConfigurationGenerator(files=[os.path.abspath(f) for f in input_files], workdir=outputdir)
    config_gen.add_remove_duplicates()
    config_gen.add_filter([{'LengthFilter': {'unit': 'word', 'min_length': 1, 'max_length': max_length}}])
    config_gen.add_subset(sample_size, 1)
    score_file = config_gen.add_score(filters)
    pre_config = config_gen.get_config()
    yaml.dump(pre_config, pathlib.Path(os.path.join(outputdir, 'config.yaml')))
    opusf = OpusFilter(pre_config)
    opusf.execute_steps(overwrite=overwrite)
    return os.path.join(outputdir, score_file)


def get_default_parameters(filter_name):
    """Get default parameters for a filter

    Uses the signature of the class. Arguments without default
    values are ignored and will cause a failure.

    """
    filter_cls = getattr(filtermodule, filter_name)
    default_parameters = {}
    sig = inspect.signature(filter_cls)
    logger.info("signature: %s%s", filter_name, sig)
    for key, parameter in sig.parameters.items():
        if parameter.default == inspect.Signature.empty:
            if key != 'kwargs':
                logger.warning("Ignoring argument without default: %s", key)
            continue
        default_parameters[key] = parameter.default
    return default_parameters


def parse_filter_specs(specs):
    """Return classname, params tuple for filter specifications"""
    if isinstance(specs, str):
        name = specs
        params = {}
    else:
        name, params = copy.deepcopy(specs)
    if '.' in name:
        name, subname = name.split('.', maxsplit=1)
        params['name'] = subname
    return name, params


class AutoFiltersABC(metaclass=abc.ABCMeta):
    """Abstract base class for automatic filter configuration"""

    DEFAULT_FILTERS = []

    def __init__(self, langs=None, scripts=None, filters=None, **kwargs):
        if filters is None:
            filters = self.DEFAULT_FILTERS
        filters = [parse_filter_specs(spec) for spec in filters]
        self.filters_to_add = []
        for filter_name, filter_params in filters:
            if filter_name == 'CharacterScoreFilter' and 'scripts' not in filter_params:
                if not scripts:
                    logger.warning('Cannot add CharacterScoreFilter (no scripts provided)')
                    continue
                filter_params['scripts'] = scripts
            if filter_name == 'LanguageIDFilter' and 'languages' not in filter_params:
                if not langs:
                    logger.warning('Cannot add LanguageIDFilter (no languages provided)')
                    continue
                filter_params['languages'] = langs
            self.filters_to_add.append((filter_name, filter_params))
        self._filters = []  # Final filters
        if kwargs:
            logger.warning("Unused arguments: %s", kwargs)

    @property
    def filters(self):
        """Get filter configuration with thresholds"""
        return self._filters

    @abc.abstractmethod
    def set_filter_thresholds(self):
        """Set filter thresholds"""


class DefaultParameterFilters(AutoFiltersABC):
    """Filter configuration with default parameters"""

    DEFAULT_FILTERS = ['LengthFilter',
                       ('LengthRatioFilter.char', {'unit': 'char'}),
                       ('LengthRatioFilter.word', {'unit': 'word'}),
                       'LongWordFilter', 'HtmlTagFilter',
                       'AverageWordLengthFilter', 'AlphabetRatioFilter',
                       'TerminalPunctuationFilter', 'NonZeroNumeralsFilter',
                       'LongestCommonSubstringFilter', 'SimilarityFilter', 'RepetitionFilter',
                       'CharacterScoreFilter', ('LanguageIDFilter', {'id_method': 'lingua'})]

    def set_filter_thresholds(self):
        """Set filter thresholds"""
        for filter_name, filter_params in self.filters_to_add:
            try:
                filter_config = self.get_filter_parameters(filter_name, filter_params)
            except FilterArgumentFailure as err:
                logger.error("Unusable default arguments for %s: %s", filter_name, err)
                continue
            self._filters.append(filter_config)

    @staticmethod
    def get_filter_parameters(filter_name, filter_params):
        """Return parameters for filter of the given class"""
        filter_cls = getattr(filtermodule, filter_name)
        defaults = get_default_parameters(filter_name)
        defaults.update(filter_params)
        try:
            filter_cls(**defaults)
        except ConfigurationError as err:
            raise FilterArgumentFailure(err) from err
        filter_config = {filter_name: defaults}
        return filter_config


class DataBasedFiltersABC(AutoFiltersABC, metaclass=abc.ABCMeta):
    """Abstract base class for filter configuration based on data"""

    def __init__(self, files, sample_size=100000, max_length=1000, inter_dir=None, overwrite=False, **kwargs):
        super().__init__(**kwargs)
        self.files = files
        self.sample_size = sample_size
        if inter_dir:
            self.use_tmp = False
            self.inter_dir = inter_dir
            if not os.path.exists(self.inter_dir):
                os.makedirs(self.inter_dir)
        else:
            self.use_tmp = True
            self.inter_dir = tempfile.mkdtemp()
        self.overwrite = overwrite
        self.max_length = max_length


class PercentileFilters(DataBasedFiltersABC):
    """Filter configuration based on filter score percentiles"""

    DEFAULT_FILTERS = ['LengthFilter',
                       ('LengthRatioFilter.char', {'unit': 'char'}),
                       ('LengthRatioFilter.word', {'unit': 'word'}),
                       'LongWordFilter', 'HtmlTagFilter',
                       'AverageWordLengthFilter', 'AlphabetRatioFilter',
                       'TerminalPunctuationFilter', 'NonZeroNumeralsFilter',
                       'LongestCommonSubstringFilter', 'SimilarityFilter', 'RepetitionFilter',
                       'CharacterScoreFilter', ('LanguageIDFilter', {'id_method': 'lingua'})]

    def __init__(self, files, excluded_percentile=0.001, **kwargs):
        super().__init__(files, **kwargs)
        self.excluded_percentile = excluded_percentile
        self.df = None

    def set_filter_thresholds(self):
        """Set filter thresholds"""
        score_file = get_score_file(
            self.files, [{name: params} for name, params in self.filters_to_add], self.inter_dir, self.sample_size,
            overwrite=self.overwrite, max_length=self.max_length)
        self.df = load_dataframe(score_file)
        for filter_name, filter_params in self.filters_to_add:
            try:
                filter_config = self.get_filter_parameters(filter_name, filter_params)
            except FilterArgumentFailure as err:
                logger.error("Unusable default arguments for %s: %s", filter_name, err)
                continue
            self._filters.append(filter_config)
        if self.use_tmp:
            shutil.rmtree(self.inter_dir)

    def get_filter_parameters(self, filter_name, filter_params):
        """Return parameters for filter of the given class"""
        adjuster = PercentileAdjuster(filter_name, filter_params)
        filter_cls = getattr(filtermodule, filter_name)
        try:
            filter_cls(**adjuster.initial_parameters)
        except ConfigurationError as err:
            raise FilterArgumentFailure(err) from err
        column_prefix = filter_name
        if 'name' in filter_params:
            column_prefix += '.' + filter_params['name']
        columns = [col for col in self.df.columns if col.startswith(column_prefix)]
        new_parameters = adjuster.get_adjusted_parameters(
            self.df[columns], excluded_percentile=self.excluded_percentile)
        filter_config = {filter_name: new_parameters}
        return filter_config


class FilterInspect:
    """Helper methods for parameters a single filter"""

    # Lists of possible filter threshold arguments
    SINGLE_THRESHOLD_ARGUMENTS = ['threshold']
    MULTI_THRESHOLD_ARGUMENTS = ['threshold', 'thresholds']
    MIN_MAX_ARGUMENTS = [('min_length', 'max_length'), ('min_threshold', 'max_threshold')]
    ALL_THRESHOLD_ARGUMENTS = SINGLE_THRESHOLD_ARGUMENTS + MULTI_THRESHOLD_ARGUMENTS + MIN_MAX_ARGUMENTS

    def __init__(self, filterclass, filter_parameters=None):
        if isinstance(filterclass, str):
            self.filter_name = filterclass
            self.filter_cls = getattr(filtermodule, self.filter_name)
        else:
            self.filter_name = filterclass.__name__
            self.filter_cls = filterclass
        self.initial_parameters = get_default_parameters(self.filter_name)
        if filter_parameters:
            self.initial_parameters.update(filter_parameters)

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
        if self._locate_arguments(self.ALL_THRESHOLD_ARGUMENTS, self.initial_parameters):
            # Known threshold parameters to adjust
            return True
        return False

    def find_threshold_keys(self, number):
        """Return threshold parameters compatible with the number of thresholds"""
        score_dir = self.filter_cls.score_direction
        if score_dir in {CLEAN_LOW, CLEAN_HIGH}:
            # Clean is below or above threshold
            if number > 1:
                # Multiple thresholds allowed (or needed)
                threshold_key = self._locate_arguments(self.MULTI_THRESHOLD_ARGUMENTS, self.initial_parameters)
            else:
                # Single threshold
                threshold_key = self._locate_arguments(self.SINGLE_THRESHOLD_ARGUMENTS, self.initial_parameters)
            if not threshold_key:
                logger.warning("Cannot find threshold parameter from %s", list(self.initial_parameters))
        elif score_dir == CLEAN_BETWEEN:
            # Clean is between minimum and maximum
            threshold_key = self._locate_arguments(self.MIN_MAX_ARGUMENTS, self.initial_parameters)
            if not threshold_key:
                logger.warning("Cannot find threshold parameter from %s", list(self.initial_parameters))
        else:
            threshold_key = None
            logger.warning("Threshold adjusting not supported for %s", self.filter_name)
        return threshold_key


class PercentileAdjuster(FilterInspect):
    """Class for setting filters to remove given percentile of data"""

    @staticmethod
    def _find_index(target, sorted_value_list):
        """Locate index of value exceeding target in sorted list

        If target is not in the list, returns the index of the first
        value exceeding it.

        """
        for idx, value2 in enumerate(sorted_value_list):
            if value2 >= target:
                return idx
        return len(sorted_value_list) - 1

    def _select_value(self, values, percentile, stats_key, reverse):
        """Select threshold value for single column"""
        sorted_values = sorted(values.unique(), reverse=reverse)
        if len(sorted_values) == 1:
            # No variation -> accept all
            return self.filter_cls.accept_threshold
        stats = values.describe(percentiles=[percentile])
        logger.info(stats)
        value = stats.loc[stats_key]
        idx = self._find_index(value, sorted_values)
        if idx < len(sorted_values) - 1:
            # Using the next value ensures that the filter will not
            # exclude everything
            value = sorted_values[idx + 1]
        return value.item()

    def _select_values(self, df, score_dir, excluded_percentile):
        """Select threshold values from the distribution in dataframe

        df: DataFrame containing the data values for the filter
        score_dir: filter's score direction (CLEAN_LOW or CLEAN_HIGH)
        excluded_percentile: target percentile to exclude as noisy

        """
        if score_dir == CLEAN_LOW:
            percentile = 1 - excluded_percentile
            stats_key = f'{100*(1-excluded_percentile):g}%'
            reverse = False
        else:
            percentile = excluded_percentile
            stats_key = f'{100*excluded_percentile:g}%'
            reverse = True
        return [self._select_value(df[column], percentile, stats_key, reverse) for column in df.columns]

    def _select_values_between(self, df, excluded_percentile):
        """Select min-max threshold values from the distribution in dataframe

        df: DataFrame containing the data values for the filter
        excluded_percentile: target percentile to exclude as noisy

        """
        half_pct = excluded_percentile / 2
        percentiles = [half_pct, 1 - half_pct]
        pct_keys = [f'{100*half_pct:g}%', f'{100*(1-half_pct):g}%']
        min_values, max_values = [], []
        for column in df.columns:
            sorted_values = sorted(df[column].unique(), reverse=False)
            if len(sorted_values) == 1:
                # No variation -> accept all
                min_value, max_value = self.filter_cls.accept_threshold
            else:
                stats = df[column].describe(percentiles=percentiles)
                logger.info(stats)
                min_value = stats.loc[pct_keys[0]].item()
                min_idx = self._find_index(min_value, sorted_values)
                if min_idx > 0:
                    # Ensure that the filter will not exclude everything
                    min_value = sorted_values[min_idx - 1].item()
                max_value = stats.loc[pct_keys[1]].item()
                max_idx = self._find_index(max_value, sorted_values)
                if max_idx < len(sorted_values) - 1:
                    # Ensure that the filter will not exclude everything
                    max_value = sorted_values[max_idx + 1].item()
            min_values.append(min_value)
            max_values.append(max_value)
        return min_values, max_values

    def get_adjusted_parameters(self, df, excluded_percentile=0.01):
        """Estimate parameters for the filter using data

        Assumes that excluded_percentile amount of the data should be
        excluded by the filter. Depending on filter's score_direction,
        they might be the lowest values, the highest values, or both
        the lowest and highest values.

        """
        parameters = copy.deepcopy(self.initial_parameters)
        if not self.is_adjustable():
            return parameters
        score_dir = self.filter_cls.score_direction
        logger.info("score type for %s: %s", self.filter_name, score_dir)
        score_dim = len(df.columns)
        threshold_key = self.find_threshold_keys(score_dim)
        if threshold_key is None:
            return parameters
        if score_dir in {CLEAN_LOW, CLEAN_HIGH}:
            # Clean is below or above threshold
            values = self._select_values(df, score_dir, excluded_percentile)
            logger.info("Selected values %s for %s", values, threshold_key)
            parameters[threshold_key] = values[0] if score_dim == 1 else values
        elif score_dir == CLEAN_BETWEEN:
            # Clean is between minimum and maximum
            min_values, max_values = self._select_values_between(df, excluded_percentile)
            if score_dim == 1:
                min_values = min_values[0]
                max_values = max_values[0]
            logger.info("Selected values %s - %s for %s", min_values, max_values, threshold_key)
            parameters[threshold_key[0]] = min_values
            parameters[threshold_key[1]] = max_values
        else:
            logger.warning("Threshold adjusting not supported")
        return parameters


class ClusterFilters(DataBasedFiltersABC):
    """Filter configuration based on score clustering"""

    DEFAULT_FILTERS = ['AlphabetRatioFilter',
                       ('LengthRatioFilter.char', {'unit': 'char'}),
                       ('LengthRatioFilter.word', {'unit': 'word'}),
                       'NonZeroNumeralsFilter',
                       'CharacterScoreFilter',
                       ('LanguageIDFilter', {'id_method': 'lingua'}),
                       'TerminalPunctuationFilter']

    def __init__(self, files, k=2, max_length=150, **kwargs):
        super().__init__(files, max_length=150, **kwargs)
        self.k = k
        self.label_file_path = os.path.join(self.inter_dir, 'labels.txt')
        self.scoredata = None

    def set_filter_thresholds(self):
        """Get filter configuration with thresholds"""
        score_file = get_score_file(
            self.files, [{name: params} for name, params in self.filters_to_add], self.inter_dir, self.sample_size,
            overwrite=self.overwrite, max_length=self.max_length)
        self.scoredata = ScoreClusters(score_file, k=self.k)
        self._set_parameters(self.scoredata.get_result_df())
        if os.path.isfile(self.label_file_path) and not self.overwrite:
            logger.info('Label file "%s" exits, not overwriting', self.label_file_path)
        else:
            with open(self.label_file_path, 'w', encoding='utf-8') as label_file:
                for label in self.scoredata.labels:
                    label_file.write(str(label)+'\n')
        if self.use_tmp:
            shutil.rmtree(self.inter_dir)

    def _set_parameters(self, df):
        """Set filter parameters based on ScoreClusters

        thresholds: list of threshold values
        rejects: boolean-valued dictionary, dataframe columns as keys

        """
        self._filters = []
        for classname, params in self.filters_to_add:
            new_params = copy.deepcopy(params)
            filter_inspect = FilterInspect(classname, new_params)
            column_prefix = classname
            if 'name' in params:
                column_prefix += '.' + params['name']
            df_part = df[df.name.str.startswith(column_prefix)]
            if all(df_part.reject):
                continue
            threshold_key = filter_inspect.find_threshold_keys(len(df_part))
            if threshold_key is None:
                continue
            thresholds = list(df_part['threshold'])
            for i, reject in enumerate(df_part.reject):
                if reject:
                    # Set a threshold that accepts all input
                    thresholds[i] = filter_inspect.filter_cls.accept_threshold
            new_params[threshold_key] = thresholds if len(thresholds) > 1 else thresholds[0]
            self._filters.append({classname: new_params})
