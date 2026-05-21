"""Hashing parallel segments"""

import collections
import functools

import regex
import xxhash

from . import ConfigurationError
from .tokenization import get_tokenize


class SegmentHasher:
    """Hasher for text segments"""

    not_letter = regex.compile(r'[^\p{L}]')
    letters = regex.compile(r'^[\p{L}]+$')
    join_char = '\n'
    join_char_escaped = r'\n'

    def __init__(self, compare='all', method='xxh64', hashseed=0, lowercase=False,
                 letters_only=False, letter_words_only=False, tokenizers=None):
        """Create a hasher for parallel segments

        Keyword arguments:
          compare -- a list of indices for selecting the segments to hash or 'all' (default 'all')
          method -- hash function from xxhash library, None for no hashing (default 'xxh64')
          hashseed -- integer seed for the hash algorithm (default 0)
          lowercase -- lowercase input strings before hashing
          letters_only -- remove all non-letters from input strings before hashing
          letter_words_only -- remove all tokens that contain non-letter characters before hashing
          tokenizers -- tokenizer specifications to use with letter_words_only, one per parallel segment

        If letter_words_only is enabled and no tokenizer specifications are provided, it is
        assumed that the inputs are pre-tokenized (words separated by whitespace).

        """
        self.compare = None
        if compare != 'all':
            if not isinstance(compare, list) or not all(isinstance(x, int) for x in compare):
                raise ConfigurationError(
                    "The compare parameter for hashing has to be 'all' or "
                    "a list of input file indices")
            self.compare = sorted(compare)
        if method == 'xx_64':
            # pyhash compability; old default method
            method = 'xxh64'
        if method and not hasattr(xxhash, method + '_intdigest'):
            raise ConfigurationError(f"Algorithm '{method}' not available from from xxhash")
        self.hashfunc = functools.partial(self._xxhash_func, method=method, seed=hashseed)
        self.lowercase = lowercase
        self.letters_only = letters_only
        self.letter_words_only = letter_words_only
        self.tokenizer_specs = tokenizers
        self._tokenizers = collections.defaultdict(lambda: None)

    @staticmethod
    def _xxhash_func(string, method, seed):
        """Return integer hash value for string given method and seed

        If no method is provided, returns the original string. Prior
        to hashing, string is encoded with utf_16_le for pyhash
        compability.

        """
        if not method:
            return string
        func = getattr(xxhash, method + '_intdigest')
        return func(string.encode('utf_16_le'), seed=seed)

    def preprocess(self, inputstr, tokenizer=None):
        """Preprocess string"""
        if self.letter_words_only:
            if tokenizer:
                inputstr = tokenizer.tokenize(inputstr)
            inputstr = ' '.join(token for token in inputstr.split() if self.letters.search(token))
        if self.letters_only:
            inputstr = regex.sub(self.not_letter, '', inputstr)
        if self.lowercase:
            inputstr = inputstr.lower()
        inputstr = inputstr.replace(self.join_char, self.join_char_escaped)
        return inputstr

    def apply(self, segments):
        """Hash a list of segments"""
        if self.tokenizer_specs:
            spec_indices = range(len(self.tokenizer_specs)) if self.compare is None else self.compare
            for idx in spec_indices:
                # Initialize tokenizers if not yet done
                if idx not in self._tokenizers:
                    self._tokenizers[idx] = get_tokenize(self.tokenizer_specs[idx])
        if self.compare is None:
            inputstr = self.join_char.join(
                self.preprocess(seg, tokenizer=self._tokenizers[idx]) for idx, seg in enumerate(segments))
        else:
            try:
                inputstr = self.join_char.join(
                    self.preprocess(segments[idx], tokenizer=self._tokenizers[idx]) for idx in self.compare)
            except KeyError as err:
                raise ConfigurationError(
                    f"The input indices {self.compare} in the compare parameter do not match input of "
                    f"length {len(segments)}") from err
        return self.hashfunc(inputstr)
