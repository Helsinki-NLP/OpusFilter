"""Corpus preprocessing"""

from functools import reduce
from itertools import zip_longest
import logging
import operator
import re

import sentence_splitter

from . import PreprocessorABC, ConfigurationError
from .tokenization import get_tokenize


logger = logging.getLogger(__name__)


class Tokenizer(PreprocessorABC):
    """Tokenize text"""

    def __init__(self, tokenizer=None, languages=None, options=None, **kwargs):
        if tokenizer is None or not isinstance(tokenizer, (str, list)):
            raise ConfigurationError("Tokenizer method(s) needs to be defined in tokenizer")
        if languages is None or not isinstance(languages, list):
            raise ConfigurationError(f"List of language code needs to be defined in languages, given {languages}")
        if options is None:
            options = {}
        if not (isinstance(options, dict) or isinstance(options, list) and isinstance(options[0], dict)):
            raise ConfigurationError(f"Options should be a dictionary or a list of dictionaries, given {options}")
        tokenizers = len(languages) * [tokenizer] if isinstance(tokenizer, str) else tokenizer
        if isinstance(options, dict):
            options = len(languages) * [options]
        if len(tokenizers) != len(languages):
            raise ConfigurationError("The number of languages does not match to the number of tokenizers")
        if options and len(options) != len(languages):
            raise ConfigurationError("The number of languages does not match to the number of tokenizer options")
        self.tokenizers = [get_tokenize((tok, lang, opt)) for tok, lang, opt in zip(tokenizers, languages, options)]
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
            lang_patterns = dict(enumerate(lang_patterns))
        self.lang_patterns = {
            idx: self._compile_patterns(idx_patterns)
            for idx, idx_patterns in lang_patterns.items()
        }
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


class MonolingualSentenceSplitter(PreprocessorABC):
    """Monolingual sentence splitter

    Uses heuristic algorithm by Philipp Koehn and Josh Schroeder
    developed for Europarl (:cite:`koehn-2005-europarl`), imported
    from the sentence-splitter library. Supports mostly European
    languages, but a non-breaking prefix file for new languages can be
    provided.

    Warning: Do not use for parallel data, as the number of lines
    (sentences) in the output may not match.

    """

    def __init__(self, language=None, non_breaking_prefix_file=None, enable_parallel=False, **kwargs):
        if language is None:
            raise ConfigurationError("A language code needs to be defined")
        if non_breaking_prefix_file:
            self.splitter = sentence_splitter.SentenceSplitter(
                language=language, non_breaking_prefix_file=non_breaking_prefix_file)
        else:
            self.splitter = sentence_splitter.SentenceSplitter(language=language)
        self.enable_parallel = enable_parallel
        super().__init__(**kwargs)

    def process(self, pairs):
        for segments in pairs:
            if len(segments) > 1 and not self.enable_parallel:
                raise ConfigurationError(
                    "MonolingualSentenceSplitter should not be used for parallel data. "
                    "To disable the error, use enable_parallel=True.")
            outputs = [self.splitter.split(segment) for segment in segments]
            for output_pair in zip_longest(*outputs, fillvalue=''):
                yield list(output_pair)
