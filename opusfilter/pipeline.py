"""Filter pipeline"""

import collections
import importlib
import logging

from tqdm import tqdm

from . import filters as filtermodule
from . import preprocessors as preprocessmodule
from . import subwords as subwordsmodule
from .util import grouper


logger = logging.getLogger(__name__)


class FilterPipeline:
    """Pipeline for combining multiple filters"""

    def __init__(self, filters=None):
        self.filters = [] if filters is None else filters
        self._chunksize = 100000

    @classmethod
    def from_config(cls, config, workdir=None):
        """Initilize filter pipeline from configuration dictionary"""
        pipeline = cls()
        for filt in config:
            custom_module = filt.pop('module') if 'module' in filt else None
            name = next(iter(filt.keys()))
            attributes = filt[name]
            if custom_module:
                mod = importlib.import_module(custom_module)
                filter_cls = getattr(mod, name)
            else:
                filter_cls = getattr(filtermodule, name)
            if workdir:
                attributes['workdir'] = workdir
            pipeline.filters.append(filter_cls(**attributes))
        return pipeline

    @property
    def chunksize(self):
        """Chunk size for score and filterfalse methods"""
        return self._chunksize

    @chunksize.setter
    def chunksize(self, value):
        if not (isinstance(value, int) and value > 0):
            raise ValueError("positive integer value required for chunksize")
        self._chunksize = value

    def get_score_tuples(self):
        """Return unique score name tuples for the filters in the pipeline"""
        fnames = [(f.__class__.__name__, f.name) for f in self.filters]
        counts = collections.Counter(fnames)
        instances = collections.Counter()
        renamed = []
        for nametuple in fnames:
            clsname, name = nametuple
            if counts[nametuple] > 1:
                instances[nametuple] += 1
                newtuple = (clsname, str(instances[nametuple])) if name is None \
                    else (clsname, name, str(instances[nametuple]))
            else:
                newtuple = (clsname, ) if name is None else (clsname, name)
            renamed.append(newtuple)
        return renamed

    def score(self, pairs):
        """Yield dictionaries of filter scores for sentence pairs"""

        def _update_score_dict(scored, namet, score):
            for key in namet[:-1]:
                if key not in scored:
                    scored[key] = {}
                scored = scored[key]
            scored[namet[-1]] = score

        fnames = self.get_score_tuples()
        for num, chunk in enumerate(grouper(pairs, self._chunksize)):
            chunk_scores = []
            for namet, filt in zip(fnames, self.filters):
                logger.info("Processing chunk %s with %s", num, '.'.join(namet))
                scorelist = list(tqdm(filt.score(chunk)))
                chunk_scores.append(scorelist)
            for scores in zip(*chunk_scores):
                output = {}
                for idx, score in enumerate(scores):
                    _update_score_dict(output, fnames[idx], score)
                yield output

    def filter(self, pairs):
        """Yield sentence pairs accepted by all filters"""
        for filt in self.filters:
            pairs = filt.filter(pairs)
        return pairs

    def filterfalse(self, pairs):
        """Yield sentence pairs rejected by any of the filters

        This is the opposite result of pipeline's filter(), and not a
        combination of filterfalse of the filters, which would yield a
        pair only if all of the filters reject it.

        """
        fnames = self.get_score_tuples()
        for num, chunk in enumerate(grouper(pairs, self._chunksize)):
            current = chunk
            remaining = []
            for namet, filt in zip(fnames, self.filters):
                logger.info("Processing chunk %s with %s (%s to check)",
                            num, '.'.join(namet), len(current))
                for pair, dec in zip(tqdm(current), filt.decisions(current)):
                    if not dec:
                        yield pair
                    else:
                        remaining.append(pair)
                if not remaining:
                    # All yielded for this chunk
                    break
                current = remaining
                remaining = []


class PreprocessorPipeline:
    """Pipeline for combining multiple preprocessors"""

    def __init__(self, preprocessors=None):
        self.preprocessors = [] if preprocessors is None else preprocessors

    @classmethod
    def from_config(cls, config, workdir=None):
        """Initilize filter pipeline from configuration dictionary"""
        pipeline = cls()
        for processor in config:
            custom_module = processor.pop('module') if 'module' in processor else None
            name = next(iter(processor.keys()))
            attributes = processor[name]
            if workdir:
                attributes['workdir'] = workdir
            if custom_module:
                mod = importlib.import_module(custom_module)
                processor_cls = getattr(mod, name)
            elif hasattr(subwordsmodule, name):
                processor_cls = getattr(subwordsmodule, name)
            else:
                processor_cls = getattr(preprocessmodule, name)
            pipeline.preprocessors.append(processor_cls(**attributes))
        return pipeline

    def process(self, pairs):
        """Yield segments processed by all preprocessors"""
        for preprocessor in self.preprocessors:
            pairs = preprocessor.process(pairs)
        return pairs
