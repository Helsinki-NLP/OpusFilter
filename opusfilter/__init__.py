"""Opusfilter package"""

import abc
import itertools
import logging


logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Configuration error for filters"""
    pass


def grouper(iterable, num):
    """Split data into fixed-length chunks"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, num))
        if not chunk:
            return
        yield chunk


class FilterABC(metaclass=abc.ABCMeta):
    """Abstract base class for sentence pair filters"""

    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs
        if kwargs:
            logging.warning("Ignoring extra keyword arguments: %s", kwargs)

    @abc.abstractmethod
    def score(self, pairs):
        """For each sentence pair, yield score(s)"""
        pass

    @abc.abstractmethod
    def accept(self, score):
        """Return filtering decision for score"""
        pass

    def decisions(self, pairs):
        """For each sentence pair, yield True if pair is accepted, False otherwise"""
        for score in self.score(pairs):
            yield self.accept(score)

    def filter(self, pairs):
        """Yield only accepted sentence pairs"""
        for sent1, sent2 in pairs:
            if self.accept(next(self.score([(sent1, sent2)]))):
                yield sent1, sent2

    def filterfalse(self, pairs):
        """Yield sentence pairs that are not accepted"""
        for sent1, sent2 in pairs:
            if not self.accept(next(self.score([(sent1, sent2)]))):
                yield sent1, sent2
