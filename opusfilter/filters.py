"""Corpus filtering"""

import logging
import string
import math
import difflib

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
        for sent1, sent2 in pairs:
            if self.unit == 'word':
                length1 = len(sent1.split())
                length2 = len(sent2.split())
            else:
                length1 = len(sent1)
                length2 = len(sent2)
            yield {'src': length1, 'tgt': length2}

    def accept(self, score):
        length1, length2 = score['src'], score['tgt']
        return (length1 >= self.min_length and length2 >= self.min_length and
                length1 <= self.max_length and length2 <= self.max_length)


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
        for sent1, sent2 in pairs:
            if self.unit == 'word':
                length1 = len(sent1.split())
                length2 = len(sent2.split())
            else:
                length1 = len(sent1)
                length2 = len(sent2)
            if length1 == 0 or length2 == 0:
                yield float('inf')
            else:
                lens = sorted([length1, length2])
                yield lens[1] / lens[0]

    def accept(self, score):
        return score < self.threshold


class LongWordFilter(FilterABC):
    """Word length filter"""

    def __init__(self, threshold=40, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def score(self, pairs):
        for sent1, sent2 in pairs:
            longest = 0
            for word in sent1.split() + sent2.split():
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
        for sent1, sent2 in pairs:
            src_tags = bool(bs(sent1, 'html.parser').find())
            tgt_tags = bool(bs(sent2, 'html.parser').find())
            yield {'src': src_tags, 'tgt': tgt_tags}

    def accept(self, score):
        src_tags, tgt_tags = score['src'], score['tgt']
        return not (src_tags or tgt_tags)


class CharacterScoreFilter(FilterABC):
    """Proportion of alphabetic characters that are in the given script

    For a list of valid scripts, see e.g.
    https://www.regular-expressions.info/unicode.html

    """

    def __init__(self, src_script='Latin', tgt_script='Latin',
                 src_threshold=1, tgt_threshold=1, **kwargs):
        self.src_script = src_script
        self.tgt_script = tgt_script
        self.src_threshold = src_threshold
        self.tgt_threshold = tgt_threshold
        self.re_not_alphas = regex.compile(r'\p{Alphabetic=No}')
        self.re_not_src_script = regex.compile(r'\p{{^Script={}}}'.format(src_script))
        self.re_not_tgt_script = regex.compile(r'\p{{^Script={}}}'.format(tgt_script))
        super().__init__(**kwargs)

    def score(self, pairs):
        for sent1, sent2 in pairs:
            scores = {}
            src_alphas = regex.sub(self.re_not_alphas, '', sent1)
            if src_alphas:
                src_script = regex.sub(self.re_not_src_script, '', src_alphas)
                scores['src'] = len(src_script) / len(src_alphas)
            else:
                scores['src'] = 1.0
            tgt_alphas = regex.sub(self.re_not_alphas, '', sent2)
            if tgt_alphas:
                tgt_script = regex.sub(self.re_not_tgt_script, '', tgt_alphas)
                scores['tgt'] = len(tgt_script) / len(tgt_alphas)
            else:
                scores['tgt'] = 1.0
            yield scores

    def accept(self, score):
        src_score, tgt_score = score['src'], score['tgt']
        return (src_score >= self.src_threshold and
                tgt_score >= self.tgt_threshold)


class LanguageIDFilter(FilterABC):
    """Language identification confidence filter"""

    def __init__(self, src_lang=None, tgt_lang=None, id_method='langid',
                 src_threshold=0, tgt_threshold=0, **kwargs):
        if not (isinstance(src_lang, str) and isinstance(tgt_lang, str)):
            logging.error("Both source and target languages need to be defined")
            raise ValueError("Strings expected, got: %s %s" % (src_lang, tgt_lang))
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.id_method = id_method
        self.src_threshold = src_threshold
        self.tgt_threshold = tgt_threshold
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
        for sent1, sent2 in pairs:
            src_score = self.confidence(sent1, self.src_lang)
            tgt_score = self.confidence(sent2, self.tgt_lang)
            yield {'src': src_score, 'tgt': tgt_score}

    def accept(self, score):
        score1, score2 = score['src'], score['tgt']
        return score1 > self.src_threshold and score2 > self.tgt_threshold


class TerminalPunctuationFilter(FilterABC):
    """Penalty score with respect to the co-occurrence of terminal
        punctuation marks"""

    def __init__(self, threshold=-2, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def score(self, pairs):
        for sent1, sent2 in pairs:
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
    """Similarity measure between numerals of the two sentences with
        zeros removed"""

    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def score(self, pairs):
        for sent1, sent2 in pairs:
            snums = [int(c) for c in sent1 if c in string.digits and c != '0']
            tnums = [int(c) for c in sent2 if c in string.digits and c != '0']
            seq = difflib.SequenceMatcher(None, snums, tnums)
            yield seq.ratio()

    def accept(self, score):
        return score >= self.threshold
