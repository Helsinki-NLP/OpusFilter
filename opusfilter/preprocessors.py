"""Corpus preprocessing"""

from functools import reduce
import operator
import re

from . import PreprocessorABC, ConfigurationError
from .tokenization import get_tokenize


class Tokenizer(PreprocessorABC):
    """Tokenize text"""

    def __init__(self, tokenizer=None, languages=None, options=None, **kwargs):
        if tokenizer is None:
            raise ConfigurationError("Tokenizer method needs to be defined in tokenizer")
        if languages is None or not isinstance(languages, list):
            raise ConfigurationError(
                "List of language code needs to be defined in languages, given %s" % languages)
        self.tokenizers = [get_tokenize((tokenizer, lang, options)) for lang in languages]
        super().__init__(**kwargs)

    def process(self, pairs):
        for segments in pairs:
            yield [self.tokenizers[idx].tokenize(segment) for idx, segment in enumerate(segments)]


class Detokenizer(Tokenizer):
    """Detokenize text"""

    def process(self, pairs):
        for segments in pairs:
            yield [self.tokenizers[idx].detokenize(segment) for idx, segment in enumerate(segments)]


class WhitespaceNormalizer(PreprocessorABC):
    """Normalize whitespace characters

    * Replace any sequences of whitespace characters with a single space
    * Remove leading and trailing whitespace

    """

    @staticmethod
    def _normalize(segment):
        segment = re.sub(r'\s+', ' ', segment)
        segment = segment.strip()
        return segment

    def process(self, pairs):
        for segments in pairs:
            yield [self._normalize(segment) for segment in segments]


class RegExpSub(PreprocessorABC):
    """Apply regular expression substitutions

    Multiple substitutions are applied in the given order. The default
    patterns are replaced with language-specific patterns when the
    corresponding index (starting from 0) is found in the
    lang_patterns dictionary. The lang_patterns argument may also be a
    list, if you e.g. want to use separate patterns for all languages.

    The substitution patterns are 4-tuples containing the regular
    expression, replacement, count (0 = substitute all) and flags
    (list of flag constants in the re library, e.g. ["I", "A"]).

    """

    def __init__(self, patterns=None, lang_patterns=None, **kwargs):
        self.patterns = self._compile_patterns(patterns) if patterns else []
        if lang_patterns is None:
            lang_patterns = {}
        elif isinstance(lang_patterns, list):
            lang_patterns = {idx: value for idx, value in enumerate(lang_patterns)}
        self.lang_patterns = {
            idx: self._compile_patterns(idx_patterns)
            for idx, idx_patterns in lang_patterns.items()
        }
        for idx, idx_patterns in lang_patterns.items():
            self.lang_patterns[idx] = self._compile_patterns(idx_patterns)
        super().__init__(**kwargs)

    @staticmethod
    def _compile_patterns(definitions):
        """Compile substitution patterns"""
        patterns = []
        for pattern, repl, count, flaglist in definitions:
            flags = reduce(operator.or_, [getattr(re, flag) for flag in flaglist], 0) if flaglist else 0
            patterns.append((re.compile(pattern, flags), repl, count))
        return patterns

    def process(self, pairs):
        for segments in pairs:
            output = []
            for idx, segment in enumerate(segments):
                patterns = self.lang_patterns.get(idx, self.patterns)
                for pattern, repl, count in patterns:
                    segment = re.sub(pattern, repl, segment, count=count)
                output.append(segment)
            yield output
