"""Corpus filtering"""

import logging
import string
import math
import difflib
import itertools

import regex
from langid.langid import LanguageIdentifier, model
import pycld2
from bs4 import BeautifulSoup as bs

from . import FilterABC, ConfigurationError
from .lm import CrossEntropyFilter
from .word_alignment import WordAlignFilter


class LengthFilter(FilterABC):
    """Sentence length filter"""

    def __init__(self, min_length=1, max_length=100, unit='word', **kwargs):
        if unit not in ('word', 'char', 'character'):
            raise ConfigurationError(
                "Unit has to be either 'word', 'char', or 'character', not '%s'" % unit)
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            if self.unit == 'word':
                lengths = [len(sent.split()) for sent in pair]
            else:
                lengths = [len(sent) for sent in pair]
            yield lengths

    def accept(self, score):
        return all(self.min_length <= length <= self.max_length for length in score)


class LengthRatioFilter(FilterABC):
    """Character length ratio"""

    def __init__(self, threshold=3, unit='word', **kwargs):
        if unit not in ('word', 'char', 'character'):
            raise ConfigurationError(
                "Unit has to be either 'word', 'char', or 'character', not '%s'" % unit)
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
            raise ConfigurationError("Mismatch in number of scripts {} and thresholds {}".format(
                len(self.scripts), len(self.thresholds)))
        self.re_not_alphas = regex.compile(r'\p{Alphabetic=No}')
        self.re_not_script = [regex.compile(r'\p{{^Script={}}}'.format(script))
                              for script in self.scripts]
        super().__init__(**kwargs)

    def score(self, pairs):
        for pair in pairs:
            if len(pair) != len(self.scripts):
                raise ValueError("Mismatch in number of scripts {} and sentences {}".format(
                    len(self.scripts), len(pair)))
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
    """Language identification confidence filter"""

    def __init__(self, languages=None, id_method='langid', thresholds=None, **kwargs):
        if languages is None:
            raise ConfigurationError("A list of language codes needs to be defined")
        self.languages = languages
        self.id_method = id_method
        self.thresholds = [0] * len(self.languages) if thresholds is None else thresholds
        self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        super().__init__(**kwargs)

    def confidence(self, sentence, lan):
        """Return confidence of the identifier"""
        if self.id_method == 'cld2':
            try:
                clddetails = pycld2.detect(sentence)
            except Exception as exp:
                clddetails = (0, 0, ((0, 'un', 0.0), 0))

            cldlan = clddetails[2][0][1]
            cldconf = round(clddetails[2][0][2]/100, 2)
            if cldlan != lan:
                cldconf = 0.0
            return cldconf

        elif self.id_method == 'langid':
            try:
                lidetails = self.identifier.classify(sentence)
            except Exception as exp:
                lidetails = ('un', 0.0)
            lilan, liconf = lidetails[0], round(lidetails[1], 2)
            if lilan != lan:
                liconf = 0.0
            return liconf

    def score(self, pairs):
        for pair in pairs:
            yield [self.confidence(sent, self.languages[idx]) for idx, sent in enumerate(pair)]

    def accept(self, score):
        return all(conf > threshold for conf, threshold in zip(score, self.thresholds))


class TerminalPunctuationFilter(FilterABC):
    """Penalty score with respect to the co-occurrence of terminal
        punctuation marks"""

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
    """Similarity measure between numerals of the two sentences with zeros removed"""

    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold
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
        return all(ratio >= self.threshold for ratio in score)
