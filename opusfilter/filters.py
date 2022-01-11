"""Corpus filtering"""

import difflib
import itertools
import logging
import math
import os
import string
from typing import Iterator, List, Tuple

import regex
from langid.langid import LanguageIdentifier, model
import pycld2
from bs4 import BeautifulSoup as bs
import fasttext

from . import FilterABC, ConfigurationError
from .lm import CrossEntropyFilter, CrossEntropyDifferenceFilter, LMClassifierFilter  # pylint: disable=W0611 # noqa: F401
from .word_alignment import WordAlignFilter  # pylint: disable=W0611 # noqa: F401


logger = logging.getLogger(__name__)


class LengthFilter(FilterABC):
    """Sentence length filter"""

    def __init__(self, min_length=1, max_length=100, unit='word', pass_empty=False, **kwargs):
        if unit not in ('word', 'char', 'character'):
            raise ConfigurationError(
                f"Unit has to be either 'word', 'char', or 'character', not '{unit}'")
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit
        self.pass_empty = pass_empty
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            if self.unit == 'word':
                lengths = [len(sent.split()) for sent in pair]
            else:
                lengths = [len(sent) for sent in pair]
            yield lengths

    def accept(self, score):
        if self.pass_empty and sum(score) == 0:
            return True
        return all(self.min_length <= length <= self.max_length for length in score)


class LengthRatioFilter(FilterABC):
    """Character length ratio"""

    def __init__(self, threshold=3, unit='word', **kwargs):
        if unit not in ('word', 'char', 'character'):
            raise ConfigurationError(
                f"Unit has to be either 'word', 'char', or 'character', not '{unit}'")
        self.threshold = threshold
        self.unit = unit
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            if self.unit == 'word':
                lengths = [len(sent.split()) for sent in pair]
            else:
                lengths = [len(sent) for sent in pair]
            lengths.sort()
            if lengths[0] == 0:
                if lengths[-1] == 0:
                    yield 0
                else:
                    yield float('inf')
            else:
                yield lengths[-1] / lengths[0]

    def accept(self, score):
        return score < self.threshold


class LongWordFilter(FilterABC):
    """Word length filter"""

    def __init__(self, threshold=40, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            longest = 0
            for word in (word for sent in pair for word in sent.split()):
                if len(word) > longest:
                    longest = len(word)
            yield longest

    def accept(self, score):
        return score < self.threshold


class AverageWordLengthFilter(FilterABC):
    """Average word length filter

    Returns zeros for empty segments. If pass_empty is true, pairs
    with only empty segments are accepted.

    """

    def __init__(self, min_length=2, max_length=20, pass_empty=False, **kwargs):
        self.min_length = min_length
        self.max_length = max_length
        self.pass_empty = pass_empty
        super().__init__(**kwargs)

    @staticmethod
    def _average_word_len(sentence):
        parts = sentence.split()
        if parts:
            return len(''.join(parts)) / len(parts)
        return 0

    def score(self, pairs):
        for pair in pairs:
            yield [self._average_word_len(sent) for sent in pair]

    def accept(self, score):
        if self.pass_empty and sum(score) == 0:
            return True
        return all(self.min_length <= length <= self.max_length for length in score)


class HtmlTagFilter(FilterABC):
    """Html tag filter"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            yield [bool(bs(sent, 'html.parser').find()) for sent in pair]

    def accept(self, score):
        return not any(score)


class CharacterScoreFilter(FilterABC):
    """Proportion of alphabetic characters that are in the given script

    For a list of valid scripts, see e.g.
    https://www.regular-expressions.info/unicode.html

    """

    def __init__(self, scripts=None, thresholds=None, **kwargs):
        if scripts is None:
            raise ConfigurationError("A list of language scripts needs to be defined")
        self.scripts = scripts
        self.thresholds = [1] * len(scripts) if thresholds is None else thresholds
        if len(self.scripts) != len(self.thresholds):
            raise ConfigurationError(
                f"Mismatch in number of scripts ({len(self.scripts)}) and thresholds ({len(self.thresholds)})")
        self.re_not_alphas = regex.compile(r'\p{Alphabetic=No}')
        self.re_not_script = [regex.compile(fr'\p{{^Script={script}}}')
                              for script in self.scripts]
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            if len(pair) != len(self.scripts):
                raise ValueError(f"Mismatch in number of scripts ({len(self.scripts)}) and sentences ({len(pair)})")
            scores = []
            for idx, sent in enumerate(pair):
                alphas = regex.sub(self.re_not_alphas, '', sent)
                if alphas:
                    script = regex.sub(self.re_not_script[idx], '', alphas)
                    scores.append(len(script) / len(alphas))
                else:
                    scores.append(1.0)
            yield scores

    def accept(self, score):
        return all(ratio >= threshold for ratio, threshold in zip(score, self.thresholds))


class LanguageIDFilter(FilterABC):
    """Language identification confidence filter

    Currently this supports three methods:
    * langid (default): see :cite:`lui-baldwin-2012-langid`
    * cld2: see https://github.com/CLD2Owners/cld2
    * fasttext: see :cite:`joulin-etal-2016-fasttext` and :cite:`joulin-etal-2017-bag`

    """

    def __init__(self, languages=None, id_method='langid', thresholds=None,
                 fasttext_model_path=None, langid_languages=None, cld2_options=None,
                 **kwargs):
        super().__init__(**kwargs)
        if languages is None:
            raise ConfigurationError("A list of language codes needs to be defined")
        # fasttext options
        if id_method == 'fasttext' and not fasttext_model_path:
            raise ConfigurationError("FastText language ID method was choosen without specifying "
                                     "any path to fasttext model")
        if id_method != 'fasttext' and fasttext_model_path:
            raise ConfigurationError("FastText language ID method was not choosen but fasttext "
                                     "path to model was set")
        self.fasttext_model = fasttext.load_model(os.path.join(self.workdir, fasttext_model_path)) \
            if id_method == 'fasttext' else None
        # langid options
        if id_method == 'langid':
            self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
            if langid_languages:
                self.identifier.set_languages(langid_languages)
        else:
            if langid_languages:
                raise ConfigurationError(
                    "langid_languages option is supported only by the method langid")
            self.identifier = None
        # cld2 options
        if id_method == 'cld2':
            self.cld2_options = cld2_options if cld2_options else {}
        else:
            if cld2_options:
                raise ConfigurationError("cld2_options is supported only by the method cld2")
            self.cld2_options = None
        # global options
        self.languages = languages
        self.id_method = id_method
        self.thresholds = [0] * len(self.languages) if thresholds is None else thresholds

    def confidence(self, sentence: str, lan: str) -> float:
        """Return confidence of the identifier"""
        if not sentence:
            # Prevent filtering empty lines
            return 1.0

        if self.id_method == 'cld2':
            try:
                clddetails = pycld2.detect(sentence, **self.cld2_options)
            except pycld2.error as err:
                logger.warning("pycld2 could not process '%s' due to: %s", sentence, err)
                clddetails = (0, 0, ((0, 'un', 0.0), 0))
            cldlan = clddetails[2][0][1]
            cldconf = round(clddetails[2][0][2]/100, 2)
            if cldlan != lan:
                cldconf = 0.0
            return cldconf

        if self.id_method == 'langid':
            lidetails = self.identifier.classify(sentence)
            lilan, liconf = lidetails[0], round(lidetails[1], 2)
            if lilan != lan:
                liconf = 0.0
            return liconf

        if self.id_method == 'fasttext':
            lang, confidence = self._fasttext_predict_lang(sentence)
            if lang != lan:
                liconf = 0.0
            else:
                liconf = confidence
            return liconf

        raise ValueError(f"Unknown language identification method '{self.id_method}'")

    def score(self, pairs: List[Tuple[str, str]]) -> Iterator[List[float]]:
        for pair in pairs:
            yield [self.confidence(sent, self.languages[idx]) for idx, sent in enumerate(pair)]

    def accept(self, score: Tuple[float, float]) -> bool:
        return all(conf > threshold for conf, threshold in zip(score, self.thresholds))

    def _fasttext_predict_lang(self, texts: List[str]) -> Tuple[str, float]:
        output = self.fasttext_model.predict(texts, k=1)
        confidence = output[1][0]
        label = output[0][0][9:]
        return label, confidence


class TerminalPunctuationFilter(FilterABC):
    """Penalty score with respect to the co-occurrence of terminal punctuation marks

    See :cite:`vazquez-etal-2019-university`

    """

    def __init__(self, threshold=-2, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("Only bilingual input supported by TerminalPunctuationFilter")
            sent1, sent2 = pair
            spun = len([c for c in sent1 if c in ['.', '?', '!', '…']])
            tpun = len([c for c in sent2 if c in ['.', '?', '!', '…']])
            score = abs(spun-tpun)
            if spun > 1:
                score += spun - 1
            if tpun > 1:
                score += tpun - 1
            score = -math.log(score + 1)
            yield score

    def accept(self, score):
        return score >= self.threshold


class NonZeroNumeralsFilter(FilterABC):
    """Similarity measure between numerals of the two sentences with zeros removed

    If require_all is True, all scores (for pairs of n segments) have
    to be equal or above the threshold; otherwise at least one the
    scores have to be equal or above the threshold. For bilingual
    input, it has no effect.

    See :cite:`vazquez-etal-2019-university`

    """

    def __init__(self, threshold=0.5, require_all=True, **kwargs):
        self.threshold = threshold
        self.require_all = require_all
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            nums = [[int(c) for c in sent if c in string.digits and c != '0']
                    for sent in pair]
            ratios = []
            for num1, num2 in itertools.combinations(nums, 2):
                seq = difflib.SequenceMatcher(None, num1, num2)
                ratios.append(seq.ratio())
            yield ratios

    def accept(self, score):
        if self.require_all:
            return all(ratio >= self.threshold for ratio in score)
        return any(ratio >= self.threshold for ratio in score)


class LongestCommonSubstringFilter(FilterABC):
    """Ratios of longest common substring to the shorter of the strings

    If require_all is True, all ratios (for pairs of n segments) have
    to be below the threshold; otherwise at least one the ratios have
    to be below the threshold. For bilingual input, it has no effect.

    """

    def __init__(self, threshold=0.9, require_all=True, **kwargs):
        self.threshold = threshold
        self.require_all = require_all
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            ratios = []
            for seq1, seq2 in itertools.combinations(pair, 2):
                seq = difflib.SequenceMatcher(isjunk=None, a=seq1, b=seq2)
                _, _, size = seq.find_longest_match(0, len(seq1), 0, len(seq2))
                minlen = min(len(seq1), len(seq2))
                ratios.append(0 if minlen == 0 else size / minlen)
            yield ratios

    def accept(self, score):
        if self.require_all:
            return all(ratio < self.threshold for ratio in score)
        return any(ratio < self.threshold for ratio in score)


class RepetitionFilter(FilterABC):
    """Filter segments with repeated content

    Filter segments with substrings of min_length to max_length
    characters that are repeated at least threshold number of times.
    The first occurrence is not counted to the threshold, i.e.,
    threshold 2 means that the substring has to occur three times.

    There may be optional space character(s) between the repeated
    strings that are not counted to the length. The repeated string
    cannot start with a whitespace character but is not limited
    otherwise.

    """

    def __init__(self, threshold=2, min_length=3, max_length=100, **kwargs):
        if threshold < 1:
            raise ConfigurationError("threshold for RepetitionFilter has to be at least one")
        if min_length < 1:
            raise ConfigurationError("min_length for RepetitionFilter has to be at least one")
        self._threshold = threshold
        self._min_length = min_length
        self._max_length = max_length
        self._regexp = self._get_regexp()
        super().__init__(**kwargs)

    @property
    def min_length(self):
        """Minimum number of characters in pattern"""
        return self._min_length

    @property
    def max_length(self):
        """Maximum number of characters in pattern"""
        return self._max_length

    @property
    def threshold(self):
        """Threshold for the number of repetitions"""
        return self._threshold

    def _get_regexp(self):
        """Return compiled regexp for finding repetitions"""
        rstring = f'(\\S.{{{self.min_length-1},{self.max_length}}}?)(?: *\\1){{{self.threshold},}}'
        return regex.compile(rstring)

    def get_repetitions(self, segment):
        """Return the number of repetitions and the repeated string

        Returns the number of repetitions and the repeated string for
        the first match of at least self.threshold number of
        repetitions. The segment may contain longer repetitions than
        the one returned. If there no matched repetitions, zero and
        None are returned.

        """
        match = self._regexp.search(segment)
        if match:
            full = match.group(0)
            repeated = match.group(1)
            return full.count(repeated) - 1, repeated
        return 0, None

    def score(self, pairs):
        for pair in pairs:
            yield [self.get_repetitions(sent)[0] for sent in pair]

    def accept(self, score):
        return all(repetitions < self.threshold for repetitions in score)
