"""Subword segmentation"""

import codecs
import functools
import logging
import math
import os
import random

from . import ConfigurationError, PreprocessorABC
from .util import file_open


logger = logging.getLogger(__name__)

LRU_MAX_SIZE = 1000000


class DummySegmentation(PreprocessorABC):
    """Base class for segmentation methods"""

    def __init__(self, reverse=False, **kwargs):
        self.reverse = reverse
        super().__init__(**kwargs)

    @staticmethod
    def get_subwords(word):
        """Return subwords for a single word (without boundary markings)"""
        return [word]

    @staticmethod
    def split(string):
        """Return input string with words splitted"""
        return string

    @staticmethod
    def join(string):
        """Return input string with words joined"""
        return string

    def process(self, pairs):
        """Split or join words depending on the reverse attribute"""
        for segments in pairs:
            yield [self.join(segment) if self.reverse else self.split(segment) for segment in segments]


class BPESegmentation(DummySegmentation):
    """Segmentation using Byte-Pair Encoding (BPE) model

    For details of the method, see :cite:`sennrich-etal-2016-neural`

    """

    def __init__(self, model, merges=-1, separator='@@', vocab=None, glossaries=None, dropout=0, **kwargs):
        super().__init__(**kwargs)
        from subword_nmt.apply_bpe import BPE
        try:
            modelfile = os.path.join(self.workdir, model)
        except TypeError as err:
            raise ConfigurationError(f"Model path expected, got: {model}") from err
        if not os.path.isfile(modelfile):
            raise ConfigurationError(f"File does not exist: {modelfile}")
        with codecs.open(modelfile, encoding='utf-8') as codes:
            if vocab is not None:
                with codecs.open(os.path.join(self.workdir, vocab), encoding='utf-8') as vocabfile:
                    self.model = BPE(codes, merges=merges, separator=separator, vocab=vocabfile, glossaries=glossaries)
            else:
                self.model = BPE(codes, merges=merges, separator=separator, vocab=None, glossaries=glossaries)
        self.dropout = dropout

    @staticmethod
    def train(datafilename, modelfilename, symbols=10000, min_frequency=2, num_workers=1):
        """Train BPE"""
        from subword_nmt.learn_bpe import learn_bpe
        with file_open(datafilename) as infile, file_open(modelfilename, 'w') as outfile:
            learn_bpe(infile, outfile, symbols, min_frequency=min_frequency, verbose=False, is_dict=False,
                      total_symbols=False, num_workers=num_workers)

    def get_subwords(self, word):
        """Return subwords for a single word (without boundary markings)"""
        subwords = self.model.segment_tokens([word], self.dropout)
        return [subword.replace(self.model.separator, '') if idx < len(subwords) - 1 else subword
                for idx, subword in enumerate(subwords)]

    def split(self, string):
        """Return input string with words splitted"""
        return self.model.process_line(string, self.dropout)

    def join(self, string):
        """Return input string with words joined"""
        return string.replace(self.model.separator + ' ', '')


@functools.lru_cache(maxsize=LRU_MAX_SIZE)
def cached_viterbi_segment(model, word, smooth, maxlen):
    """Cache Viterbi segmentations from Morfessor model"""
    return model.viterbi_segment(word, smooth, maxlen)


class MorfessorSegmentation(DummySegmentation):
    """Segmentation using Morfessor model

    For details of the method, see :cite:`virpioja-etal-2013-morfessor`

    """

    def __init__(self, model, separator='@@ ', lowercase=False, viterbi_max_len=30, viterbi_smoothing=0, **kwargs):
        super().__init__(**kwargs)
        from morfessor.io import MorfessorIO
        self.morfessor_io = MorfessorIO(
            encoding='utf-8', construction_separator=separator, comment_start='#', compound_separator=r'\s+',
            atom_separator=None, lowercase=lowercase)
        try:
            modelfile = os.path.join(self.workdir, model)
        except TypeError as err:
            raise ConfigurationError(f"Model path expected, got: {model}") from err
        if not os.path.isfile(modelfile):
            raise ConfigurationError(f"File does not exist: {modelfile}")
        self.model = self.morfessor_io.read_binary_model_file(modelfile)
        self.max_len = viterbi_max_len
        self.addcount = viterbi_smoothing

    @classmethod
    def train(cls, datafilename, modelfilename, lowercase=False, min_frequency=1, dampening='log',
              seed=None, use_skips=True, **kwargs):
        """Train Morfessor"""
        import morfessor
        if seed is not None:
            random.seed(seed)
        if not dampening or dampening == 'none':
            count_modifier = None
        elif dampening == 'log':
            count_modifier = lambda x: int(round(math.log(x + 1, 2)))  # noqa: E731
        elif dampening in {'ones', 'types'}:
            count_modifier = lambda x: 1  # noqa: E731
        else:
            raise ConfigurationError(f"Invalid option for Morfessor frequency dampening: {dampening}")
        morfessor_io = morfessor.io.MorfessorIO(
            encoding='utf-8', construction_separator='@@ ', comment_start='#', compound_separator=r'\s+',
            atom_separator=None, lowercase=lowercase)
        data = morfessor_io.read_corpus_file(datafilename)
        model = morfessor.baseline.BaselineModel(use_skips=use_skips, **kwargs)
        model.load_data(data, freqthreshold=min_frequency, count_modifier=count_modifier)
        model.train_batch()
        morfessor_io.write_binary_model_file(modelfilename, model)

    def get_subwords(self, word):
        """Return subwords for a single word (without boundary markings)"""
        if self.morfessor_io.lowercase:
            word = word.lower()
        return cached_viterbi_segment(self.model, word, self.addcount, self.max_len)[0]

    def split(self, string):
        """Return input string with words splitted"""
        return ' '.join(
            self.morfessor_io.format_constructions(
                self.get_subwords(word), csep=self.morfessor_io.construction_separator)
            for word in string.split())

    def join(self, string):
        """Return input string with words joined"""
        return string.replace(self.morfessor_io.construction_separator, '')
