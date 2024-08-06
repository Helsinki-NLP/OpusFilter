"""Corpus filtering"""

import difflib
import itertools
import logging
import math
import os
import string
from typing import Iterator, List, Tuple

import regex

from . import FilterABC, ConfigurationError, CLEAN_LOW, CLEAN_HIGH, CLEAN_BETWEEN, CLEAN_TRUE, CLEAN_FALSE
from .lm import CrossEntropyFilter, CrossEntropyDifferenceFilter, LMClassifierFilter  # pylint: disable=W0611 # noqa: F401
from .util import check_args_compability
from .word_alignment import WordAlignFilter      # pylint: disable=W0611 # noqa: F401
from .embeddings import SentenceEmbeddingFilter  # pylint: disable=W0611 # noqa: F401


logger = logging.getLogger(__name__)


class LengthFilter(FilterABC):
    """Sentence length filter"""

    score_direction = CLEAN_BETWEEN
    accept_threshold = (0, math.inf)
    reject_threshold = (math.inf, 0)

    def __init__(self, min_length=1, max_length=100, unit='word', pass_empty=False, **kwargs):
        min_length, max_length, unit = check_args_compability(
            min_length, max_length, unit,
            required_types=[(int, float), (int, float), str],
            choices=[None, None, ('word', 'char', 'character')],
            names=['min_length', 'max_length', 'unit'])
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit
        self.pass_empty = pass_empty
        super().__init__(**kwargs)

    def get_length(self, segment, idx):
        """Return length of the segment in index"""
        if self.unit[idx] == 'word':
            return len(segment.split())
        return len(segment)

    def score(self, pairs):
        for pair in pairs:
            yield [self.get_length(segment, idx) for idx, segment in enumerate(pair)]

    def accept(self, score):
        if self.pass_empty and sum(score) == 0:
            return True
        return all(self.min_length[idx] <= length <= self.max_length[idx] for idx, length in enumerate(score))


class LengthRatioFilter(FilterABC):
    """Character length ratio"""

    score_direction = CLEAN_LOW
    accept_threshold = math.inf
    reject_threshold = 1

    def __init__(self, threshold=3, unit='word', **kwargs):
        self.threshold = threshold
        self.unit = check_args_compability(
            unit, required_types=[str], choices=[('word', 'char', 'character')], names=['unit'])
        super().__init__(**kwargs)

    def get_length(self, segment, idx):
        """Return length of the segment in index"""
        if self.unit[idx] == 'word':
            return len(segment.split())
        return len(segment)

    def score(self, pairs):
        for pair in pairs:
            lengths = sorted(self.get_length(segment, idx) for idx, segment in enumerate(pair))
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

    score_direction = CLEAN_LOW
    accept_threshold = math.inf
    reject_threshold = 1

    def __init__(self, threshold=40, **kwargs):
        self.threshold = check_args_compability(threshold, required_types=[(int, float)], names=['threshold'])
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            yield [max((len(word) for word in segment.split()), default=0) for segment in pair]

    def accept(self, score):
        return all(length < self.threshold[idx] for idx, length in enumerate(score))


class AverageWordLengthFilter(FilterABC):
    """Average word length filter

    Returns zeros for empty segments. If pass_empty is true, pairs
    with only empty segments are accepted.

    """

    score_direction = CLEAN_BETWEEN
    accept_threshold = (0, math.inf)
    reject_threshold = (math.inf, 0)

    def __init__(self, min_length=2, max_length=20, pass_empty=False, **kwargs):
        min_length, max_length = check_args_compability(
            min_length, max_length, required_types=[(int, float), (int, float)], names=['min_length', 'max_length'])
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
        return all(self.min_length[idx] <= length <= self.max_length[idx] for idx, length in enumerate(score))


class HtmlTagFilter(FilterABC):
    """HTML tag filter"""

    score_direction = CLEAN_FALSE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def check(segment):
        """Return whether segment has HTML tags (or something that breaks bs4)"""
        import bs4
        try:
            found = bool(bs4.BeautifulSoup(segment, 'html.parser').find())
        except (TypeError, UnboundLocalError, NotImplementedError,
                AssertionError, bs4.builder.ParserRejectedMarkup) as err:
            logger.warning("BeautifulSoup parsing failed for %s: %s", repr(segment), err)
            found = True
        return found

    def score(self, pairs):
        for pair in pairs:
            yield [self.check(sent) for sent in pair]

    def accept(self, score):
        return not any(score)


class RegExpFilter(FilterABC):
    """Filter out segments that match or do not match a regular expression

    You can either provide a single regexp or one for each language in
    the parallel data. The regex library is used for the search.

    If accept_match is False, the pair is accepted only if none of the
    segment match the corresponding regexp. If accept_match is True,
    the pair is accepted only if all segments match the corresponding
    regexp.

    """

    def __init__(self, regexps=None, accept_match=False, **kwargs):
        self.regexps = check_args_compability(regexps, required_types=[str], names=['regexps'])
        self.accept_match = accept_match
        super().__init__(**kwargs)

    @property
    def score_direction(self):
        """Hint for which score values indicate accept"""
        return CLEAN_TRUE if self.accept_match else CLEAN_FALSE

    def score(self, pairs):
        for pair in pairs:
            yield [bool(regex.search(self.regexps[idx], segment)) for idx, segment in enumerate(pair)]

    def accept(self, score):
        if self.accept_match:
            return all(score)
        return not any(score)


class AlphabetRatioFilter(FilterABC):
    """Proportion of alphabetic characters in the segment"""

    score_direction = CLEAN_HIGH
    accept_threshold = 0
    reject_threshold = 1 + 10**-6

    def __init__(self, threshold=0.75, exclude_whitespace=False, **kwargs):
        self.threshold = check_args_compability(threshold, required_types=[(float, int)], names=['threshold'])
        self.exclude_whitespace = exclude_whitespace
        self.re_whitespace = regex.compile(r'\s')
        self.re_not_alphas = regex.compile(r'\p{Alphabetic=No}')
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            scores = []
            for segment in pair:
                if self.exclude_whitespace:
                    segment = self.re_whitespace.sub('', segment)
                alphas = self.re_not_alphas.sub('', segment)
                if segment:
                    scores.append(len(alphas) / len(segment))
                else:
                    scores.append(1.0)
            yield scores

    def accept(self, score):
        return all(ratio >= threshold for ratio, threshold in zip(score, self.threshold))


class CharacterScoreFilter(FilterABC):
    """Proportion of alphabetic characters that are in the given script

    For a list of valid scripts, see e.g.
    https://www.regular-expressions.info/unicode.html

    """

    score_direction = CLEAN_HIGH
    accept_threshold = 0
    reject_threshold = 1 + 10**-6

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
                alphas = self.re_not_alphas.sub('', sent)
                if alphas:
                    script = self.re_not_script[idx].sub('', alphas)
                    scores.append(len(script) / len(alphas))
                else:
                    scores.append(1.0)
            yield scores

    def accept(self, score):
        return all(ratio >= threshold for ratio, threshold in zip(score, self.thresholds))


class LanguageIDFilter(FilterABC):
    """Language identification confidence filter

    Currently this supports four methods:
    * langid (default): see :cite:`lui-baldwin-2012-langid`
    * cld2: see https://github.com/CLD2Owners/cld2
    * fasttext: see :cite:`joulin-etal-2016-fasttext` and :cite:`joulin-etal-2017-bag`
    * lingua: see https://github.com/pemistahl/lingua-py

    """

    score_direction = CLEAN_HIGH
    accept_threshold = -1
    reject_threshold = 1

    def __init__(self, languages=None, id_method='langid', thresholds=None,
                 fasttext_model_path=None, langid_languages=None, cld2_options=None,
                 lingua_mode=None, **kwargs):
        super().__init__(**kwargs)
        if languages is None:
            raise ConfigurationError("A list of language codes needs to be defined")
        self.identifier = None
        self.cld2_options = None
        self.fasttext_model = None
        self.lingua_detector = None
        if id_method == 'fasttext':
            self.init_fastttext(fasttext_model_path)
        else:
            if fasttext_model_path:
                raise ConfigurationError("FastText language ID method was not choosen but fasttext "
                                         "path to model was set")
        if id_method == 'langid':
            self.init_langid(langid_languages)
        else:
            if langid_languages:
                raise ConfigurationError(
                    "langid_languages option is supported only by the method langid")
        if id_method == 'cld2':
            self.cld2_options = cld2_options if cld2_options else {}
        else:
            if cld2_options:
                raise ConfigurationError("cld2_options is supported only by the method cld2")
        if id_method == "lingua":
            self.init_lingua(lingua_mode if lingua_mode else 'low')
        else:
            if lingua_mode:
                raise ConfigurationError("lingua_mode is supported only by the method lingua")
        # global options
        self.languages = languages
        self.id_method = id_method
        self.thresholds = [0] * len(self.languages) if thresholds is None else thresholds

    def init_langid(self, langid_languages):
        """Initialize langid identifier"""
        from py3langid.langid import LanguageIdentifier, MODEL_FILE
        self.identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
        if langid_languages:
            self.identifier.set_languages(langid_languages)

    def init_fastttext(self, fasttext_model_path):
        """Initialize fasttext identifier"""
        if not fasttext_model_path:
            raise ConfigurationError("FastText language ID method was choosen without specifying "
                                     "any path to fasttext model")
        try:
            import fasttext
        except ImportError:
            logger.warning("Could not import fasttext. Select another id_method for LanguageIDFilter.")
            raise
        self.fasttext_model = fasttext.load_model(os.path.join(self.workdir, fasttext_model_path))

    def init_lingua(self, lingua_mode):
        """Initialize lingua identifier"""
        from lingua import LanguageDetectorBuilder
        # TODO: support lingua_languages just like langid_languages
        from_languages = LanguageDetectorBuilder.from_all_languages()
        if lingua_mode == "high":
            self.lingua_detector = from_languages.with_preloaded_language_models().build()
        elif lingua_mode == "low":
            self.lingua_detector = from_languages.with_low_accuracy_mode().build()
        else:
            raise ConfigurationError(f"lingua mode '{lingua_mode}' is not supported.")

    def confidence(self, sentence: str, lan: str) -> float:
        """Return confidence of the identifier"""
        if not sentence:
            # Prevent filtering empty lines
            return 1.0

        if self.id_method == 'cld2':
            try:
                import pycld2
            except ImportError:
                logger.warning("Could not import pycld2. Select another id_method for LanguageIDFilter.")
                raise
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
            lilan, liconf = lidetails[0], round(float(lidetails[1]), 2)
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

        if self.id_method == 'lingua':
            confidence_values = self.lingua_detector.compute_language_confidence_values(sentence)
            lang = confidence_values[0].language
            confidence = confidence_values[0].value
            if lang.iso_code_639_1.name.lower() != lan:
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

    score_direction = CLEAN_HIGH
    accept_threshold = 0
    reject_threshold = math.inf

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
            score = abs(spun - tpun)
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

    score_direction = CLEAN_HIGH
    accept_threshold = 0
    reject_threshold = 1 + 10**-6

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

    score_direction = CLEAN_LOW
    accept_threshold = 1 + 10**-6
    reject_threshold = 0

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


class SimilarityFilter(FilterABC):
    """Filter on string/sequence similarity

    Uses Levenshtein distance implemented in the RapidFuzz library.
    The weights parameter can be used to change the costs of the three
    operations (insertion, deletion, substitution).

    If require_all is True, all ratios (for pairs of n segments) have
    to be below the threshold; otherwise at least one the ratios have
    to be below the threshold. For bilingual input, it has no effect.

    """

    score_direction = CLEAN_LOW
    accept_threshold = 1 + 10**-6
    reject_threshold = 0

    VALID_UNITS = ('word', 'char', 'character')

    def __init__(self, threshold=0.9, weights=(1, 1, 1), unit='char', lowercase=False,
                 require_all=True, **kwargs):
        if unit not in self.VALID_UNITS:
            raise ConfigurationError(
                f"Value of 'unit' is not one of the allowed choices {self.VALID_UNITS}: {unit}")
        self.threshold = threshold
        self.weights = weights
        self.unit = unit
        self.lowercase = lowercase
        self.require_all = require_all
        super().__init__(**kwargs)

    def similarity(self, seq1, seq2):
        """Return normalized similarity between the sequences"""
        import rapidfuzz
        if self.lowercase:
            seq1 = seq1.lower()
            seq2 = seq2.lower()
        if self.unit == 'word':
            seq1 = seq1.split()
            seq2 = seq2.split()
        return rapidfuzz.distance.Levenshtein.normalized_similarity(
            seq1, seq2, weights=self.weights)

    def score(self, pairs):
        for pair in pairs:
            yield [self.similarity(seq1, seq2) for seq1, seq2 in itertools.combinations(pair, 2)]

    def accept(self, score):
        if self.require_all:
            return all(ratio < self.threshold for ratio in score)
        return any(ratio < self.threshold for ratio in score)


class RepetitionFilter(FilterABC):
    """Filter segments with repeated content

    Filter out segments with substrings of min_length to max_length
    characters that are repeated at least threshold number of times.
    The first occurrence is not counted to the threshold, i.e.,
    threshold 2 means that any substring cannot occur three times
    in a row.

    There may be optional space character(s) between the repeated
    strings that are not counted to the length. The repeated string
    cannot start with a whitespace character but is not limited
    otherwise.

    """

    score_direction = CLEAN_LOW
    accept_threshold = math.inf
    reject_threshold = 1

    def __init__(self, threshold=2, min_length=3, max_length=100, **kwargs):
        if min_length < 1:
            raise ConfigurationError(f"min_length for RepetitionFilter has to be at least one, got {min_length}")
        if threshold < self.reject_threshold:
            raise ConfigurationError(f"threshold for RepetitionFilter has to be at least one, got {threshold}")
        self._threshold = threshold
        self._min_length = min_length
        self._max_length = max_length
        if threshold == self.accept_threshold:
            logger.warning("threshold for RepetitionFilter set to %s, filter disabled", threshold)
            self._regexp = None
        else:
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
        if not self._regexp:
            return 0, None
        match = self._regexp.search(segment)
        if match:
            full = match.group(0)
            repeated = match.group(1)
            return full.count(repeated) - 1, repeated
        return 0, None

    def score(self, pairs):
        for pair in pairs:
            yield max(self.get_repetitions(sent)[0] for sent in pair)

    def accept(self, score):
        return score < self.threshold
