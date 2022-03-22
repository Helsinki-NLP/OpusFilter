"""Opusfilter package"""

import abc
import logging


logger = logging.getLogger(__name__)


class OpusFilterError(Exception):
    """OpusFilter error"""


class ConfigurationError(OpusFilterError):
    """Configuration error for OpusFilter"""


class OpusFilterRuntimeError(OpusFilterError):
    """Runtime error for OpusFilter"""


class FilterABC(metaclass=abc.ABCMeta):
    """Abstract base class for sentence pair filters

    If the filter uses or creates any non-temporary files, they should
    be located under workdir.

    """

    def __init__(self, name=None, workdir='', **kwargs):
        self.name = name
        self.workdir = workdir
        self.kwargs = kwargs
        if kwargs:
            logging.warning("Ignoring extra keyword arguments: %s", kwargs)

    @abc.abstractmethod
    def score(self, pairs):
        """For each sentence pair, yield score(s)"""

    @abc.abstractmethod
    def accept(self, score):
        """Return filtering decision for score"""

    def decisions(self, pairs):
        """For each sentence pair, yield True if pair is accepted, False otherwise"""
        for score in self.score(pairs):
            yield self.accept(score)

    def filter(self, pairs):
        """Yield only accepted sentence pairs"""
        for pair in pairs:
            if self.accept(next(self.score([pair]))):  # pylint: disable=R1708
                yield pair

    def filterfalse(self, pairs):
        """Yield sentence pairs that are not accepted"""
        for pair in pairs:
            if not self.accept(next(self.score([pair]))):  # pylint: disable=R1708
                yield pair


class PreprocessorABC(metaclass=abc.ABCMeta):
    """Abstract base class for preprocessors"""

    def __init__(self, workdir='', **kwargs):
        self.workdir = workdir
        self.kwargs = kwargs
        if kwargs:
            logging.warning("Ignoring extra keyword arguments: %s", kwargs)

    @abc.abstractmethod
    def process(self, pairs):
        """For each tuple of parallel segments, yield preprocessed segments"""
