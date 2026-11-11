"""Filter pipeline"""

import collections
import logging

from tqdm import tqdm

from . import filters as filtermodule
from . import preprocessors as preprocessmodule
from . import subwords as subwordsmodule
from .util import grouper, import_class


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
            name, filter_cls = import_class(filt, [filtermodule])
            attributes = filt[name]
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

    @staticmethod
    def _update_score_dict(scored, namet, score, decision=None):
        for key in namet[:-1]:
            if key not in scored:
                scored[key] = {}
            scored = scored[key]
        if decision is not None:
            scored[namet[-1]] = {'scores': score, 'accept': decision}
        else:
            scored[namet[-1]] = score

    def _make_score_dict(self, fnames, scores):
        output = {}
        for namet, score in zip(fnames, scores):
            self._update_score_dict(output, namet, score)
        return output

    def score(self, pairs, with_decision=False):
        """Yield dictionaries of filter scores for sentence pairs"""
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
                    decision = self.filters[idx].accept(score) if with_decision else None
                    self._update_score_dict(output, fnames[idx], score, decision=decision)
                yield output

    def filter(self, pairs):
        """Yield sentence pairs accepted by all filters"""
        for filt in self.filters:
            pairs = filt.filter(pairs)
        return pairs

    def filter_with_scores(self, pairs, filterfalse=False):
        """Yield accepted pairs together with their filter scores

        Filters are applied in order and pairs rejected by a filter are
        not passed to the following filters. The score dictionaries are
        built only for pairs that are yielded. With ``filterfalse``,
        rejected pairs contain scores only from the filters that were
        evaluated before rejection.
        """
        fnames = self.get_score_tuples()
        for num, chunk in enumerate(grouper(pairs, self._chunksize)):
            current = [(pair, []) for pair in chunk]
            for idx, (namet, filt) in enumerate(zip(fnames, self.filters)):
                logger.info("Processing chunk %s with %s (%s to check)",
                            num, '.'.join(namet), len(current))
                chunk_pairs = [pair for pair, _ in current]
                scorelist = list(tqdm(filt.score(chunk_pairs)))
                remaining = []
                for (pair, pair_scores), score in zip(current, scorelist):
                    pair_scores.append(score)
                    if filt.accept(score):
                        remaining.append((pair, pair_scores))
                    elif filterfalse:
                        yield pair, self._make_score_dict(fnames[:idx + 1], pair_scores)
                current = remaining
                if not current:
                    break
            if not filterfalse:
                for pair, pair_scores in current:
                    yield pair, self._make_score_dict(fnames, pair_scores)

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
            name, processor_cls = import_class(processor, [subwordsmodule, preprocessmodule])
            attributes = processor[name]
            if workdir:
                attributes['workdir'] = workdir
            pipeline.preprocessors.append(processor_cls(**attributes))
        return pipeline

    def process(self, pairs):
        """Yield segments processed by all preprocessors"""
        for preprocessor in self.preprocessors:
            pairs = preprocessor.process(pairs)
        return pairs
