"""Language identification filters"""

import os
import logging
from typing import Iterator, List, Tuple

from iso639 import Lang
import numpy as np

from . import FilterABC, ConfigurationError, CLEAN_LOW, CLEAN_HIGH


logger = logging.getLogger(__name__)


class LangidFilter(FilterABC):
    """Language identification confidence filter based on langid

    For the description of the method, see :cite:`lui-baldwin-2012-langid`

    """

    score_direction = CLEAN_HIGH
    accept_threshold = -1
    reject_threshold = 1

    def __init__(self, languages=None, thresholds=None, langid_languages=None, **kwargs):
        super().__init__(**kwargs)
        if languages is None:
            raise ConfigurationError("A list of language codes needs to be defined")
        self.identifier = None
        from py3langid.langid import LanguageIdentifier, MODEL_FILE
        self.identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
        if langid_languages:
            self.identifier.set_languages(langid_languages)
        # global options
        self.languages = languages
        self.thresholds = [0] * len(self.languages) if thresholds is None else thresholds

    def confidence(self, sentence: str, lan: str) -> float:
        """Return confidence of the identifier"""
        if not sentence:
            # Prevent filtering empty lines
            return 1.0
        lidetails = self.identifier.classify(sentence)
        lilan, liconf = lidetails[0], round(float(lidetails[1]), 2)
        if lilan != lan:
            liconf = 0.0
        return liconf

    def score(self, pairs: List[Tuple[str, str]]) -> Iterator[List[float]]:
        for pair in pairs:
            yield [self.confidence(sent, self.languages[idx]) for idx, sent in enumerate(pair)]

    def accept(self, score: Tuple[float, float]) -> bool:
        return all(conf > threshold for conf, threshold in zip(score, self.thresholds))


class Cld2Filter(FilterABC):
    """Language identification confidence filter based on cld2

    For the description of the method, see https://github.com/CLD2Owners/cld2

    """

    score_direction = CLEAN_HIGH
    accept_threshold = -1
    reject_threshold = 1

    def __init__(self, languages=None, thresholds=None, options=None, **kwargs):
        super().__init__(**kwargs)
        if languages is None:
            raise ConfigurationError("A list of language codes needs to be defined")
        self.options = options if options else {}
        # global options
        self.languages = languages
        self.thresholds = [0] * len(self.languages) if thresholds is None else thresholds

    def confidence(self, sentence: str, lan: str) -> float:
        """Return confidence of the identifier"""
        if not sentence:
            # Prevent filtering empty lines
            return 1.0
        try:
            import pycld2
        except ImportError:
            logger.warning("Could not import pycld2")
            raise
        try:
            clddetails = pycld2.detect(sentence, **self.options)
        except pycld2.error as err:
            logger.warning("pycld2 could not process '%s' due to: %s", sentence, err)
            clddetails = (0, 0, ((0, 'un', 0.0), 0))
        cldlan = clddetails[2][0][1]
        cldconf = round(clddetails[2][0][2]/100, 2)
        if cldlan != lan:
            cldconf = 0.0
        return cldconf

    def score(self, pairs: List[Tuple[str, str]]) -> Iterator[List[float]]:
        for pair in pairs:
            yield [self.confidence(sent, self.languages[idx]) for idx, sent in enumerate(pair)]

    def accept(self, score: Tuple[float, float]) -> bool:
        return all(conf > threshold for conf, threshold in zip(score, self.thresholds))


class FastTextFilter(FilterABC):
    """Language identification confidence filter based on fasttext

    For the description of the method, see :cite:`joulin-etal-2016-fasttext` and :cite:`joulin-etal-2017-bag`

    """

    score_direction = CLEAN_HIGH
    accept_threshold = -1
    reject_threshold = 1

    def __init__(self, languages=None, thresholds=None, model_path=None, **kwargs):
        super().__init__(**kwargs)
        if languages is None:
            raise ConfigurationError("A list of language codes needs to be defined")
        try:
            import fasttext
        except ImportError:
            logger.warning("Could not import fasttext")
            raise
        if not model_path:
            raise ConfigurationError("FastTextFilter requires model_path pointing to a fasttext model")
        self.fasttext_model = fasttext.load_model(os.path.join(self.workdir, model_path))
        # global options
        self.languages = languages
        self.thresholds = [0] * len(self.languages) if thresholds is None else thresholds

    def confidence(self, sentence: str, lan: str) -> float:
        """Return confidence of the identifier"""
        if not sentence:
            # Prevent filtering empty lines
            return 1.0
        lang, confidence = self._fasttext_predict_lang(sentence)
        if lang != lan:
            liconf = 0.0
        else:
            liconf = confidence
        return liconf

    def _fasttext_predict_lang(self, texts: List[str]) -> Tuple[str, float]:
        output = self.fasttext_model.predict(texts, k=1)
        confidence = output[1][0]
        label = output[0][0][9:]
        return label, confidence

    def score(self, pairs: List[Tuple[str, str]]) -> Iterator[List[float]]:
        for pair in pairs:
            yield [self.confidence(sent, self.languages[idx]) for idx, sent in enumerate(pair)]

    def accept(self, score: Tuple[float, float]) -> bool:
        return all(conf > threshold for conf, threshold in zip(score, self.thresholds))


class LinguaFilter(FilterABC):
    """Language identification confidence filter based on Lingua

    For the description of the method, see https://github.com/pemistahl/lingua-py

    """

    score_direction = CLEAN_HIGH
    accept_threshold = -1
    reject_threshold = 1

    def __init__(self, languages=None, thresholds=None, langid_languages=None, lingua_mode='low', **kwargs):
        super().__init__(**kwargs)
        if languages is None:
            raise ConfigurationError("A list of language codes needs to be defined")
        from lingua import LanguageDetectorBuilder, IsoCode639_1
        if langid_languages:
            for code in langid_languages:
                if not hasattr(IsoCode639_1, code.upper()):
                    raise ConfigurationError(f"Language {code} not supported by lingua")
            from_languages = LanguageDetectorBuilder.from_iso_codes_639_1(
                *[getattr(IsoCode639_1, code.upper()) for code in langid_languages])
        else:
            from_languages = LanguageDetectorBuilder.from_all_languages()
        if lingua_mode == "high":
            self.lingua_detector = from_languages.with_preloaded_language_models().build()
        elif lingua_mode == "low":
            self.lingua_detector = from_languages.with_low_accuracy_mode().build()
        else:
            raise ConfigurationError(f"lingua mode '{lingua_mode}' is not supported.")
        # global options
        self.languages = languages
        self.thresholds = [0] * len(self.languages) if thresholds is None else thresholds

    def confidence(self, sentence: str, lan: str) -> float:
        """Return confidence of the identifier"""
        if not sentence:
            # Prevent filtering empty lines
            return 1.0
        confidence_values = self.lingua_detector.compute_language_confidence_values(sentence)
        lang = confidence_values[0].language
        confidence = confidence_values[0].value
        if lang.iso_code_639_1.name.lower() != lan:
            liconf = 0.0
        else:
            liconf = confidence
        return liconf

    def score(self, pairs: List[Tuple[str, str]]) -> Iterator[List[float]]:
        for pair in pairs:
            yield [self.confidence(sent, self.languages[idx]) for idx, sent in enumerate(pair)]

    def accept(self, score: Tuple[float, float]) -> bool:
        return all(conf > threshold for conf, threshold in zip(score, self.thresholds))


class HeliportSimpleFilter(FilterABC):
    """Language identification filter based on HeLI-OTS

    See :cite:`jauhiainen-etal-2022-heli` and https://github.com/ZJaume/heliport

    """

    score_direction = CLEAN_HIGH
    accept_threshold = -1
    reject_threshold = 1

    def __init__(self, languages=None, thresholds=None, **kwargs):
        super().__init__(**kwargs)
        if languages is None:
            raise ConfigurationError("A list of language codes needs to be defined")
        from heliport import Identifier
        self.identifier = Identifier()
        # global options
        self.languages = languages
        self.thresholds = [0] * len(self.languages) if thresholds is None else thresholds

    def confidence(self, sentence: str, lan: str) -> float:
        """Return simple confidence score

        Returns 1 if the predicted language is expected, 0.5 if
        undefined, and 0 if some other language.

        """
        if not sentence:
            # Prevent filtering empty lines
            return 1.0
        iso_code_639_3 = self.identifier.identify(sentence, ignore_confidence=False)
        if iso_code_639_3 == 'und':
            # special label for too low confidence; should be considered as a possible match
            return 0.5
        lang = Lang(iso_code_639_3).pt1  # convert to ISO 639-1
        if lang != lan:
            return 0.0
        return 1.0

    def score(self, pairs: List[Tuple[str, str]]) -> Iterator[List[float]]:
        for pair in pairs:
            yield [self.confidence(sent, self.languages[idx]) for idx, sent in enumerate(pair)]

    def accept(self, score: Tuple[float, float]) -> bool:
        return all(conf > threshold for conf, threshold in zip(score, self.thresholds))


class HeliportConfidenceFilter(HeliportSimpleFilter):
    """Language identification confidence based on HeLI-OTS confidence scores

    See :cite:`jauhiainen-etal-2022-heli` and https://github.com/ZJaume/heliport

    """

    score_direction = CLEAN_HIGH
    accept_threshold = -1
    reject_threshold = 7

    def confidence(self, sentence: str, lan: str) -> float:
        """Return confidence of the identifier"""
        if not sentence:
            # Prevent filtering empty lines
            return self.reject_threshold
        iso_code_639_3, score = self.identifier.identify_with_score(sentence, ignore_confidence=False)
        if iso_code_639_3 == 'und':
            # special label for too low confidence; should be considered as a possible match
            return score
        lang = Lang(iso_code_639_3).pt1  # convert to ISO 639-1
        if lang != lan:
            return 0.0
        return score

    def score(self, pairs: List[Tuple[str, str]]) -> Iterator[List[float]]:
        for pair in pairs:
            yield [self.confidence(sent, self.languages[idx]) for idx, sent in enumerate(pair)]

    def accept(self, score: Tuple[float, float]) -> bool:
        return all(conf > threshold for conf, threshold in zip(score, self.thresholds))


class HeliportRawScoreFilter(HeliportSimpleFilter):
    """Language identification filter based on raw HeLI-OTS scores

    See :cite:`jauhiainen-etal-2022-heli` and https://github.com/ZJaume/heliport

    """

    score_direction = CLEAN_LOW
    penalty_value = 7
    accept_threshold = penalty_value + 1e-6
    reject_threshold = 0

    def __init__(self, languages=None, thresholds=None, topk=50, **kwargs):
        super().__init__(languages=languages, thresholds=thresholds, **kwargs)
        if not isinstance(topk, int):
            raise ConfigurationError("Heliport topk argument must be integer.")
        self.topk = topk

    def raw_score(self, sentence: str, lan: str) -> float:
        """Return raw score of the identifier"""
        if not sentence:
            # Prevent filtering empty lines
            return 0
        results = self.identifier.identify_topk_with_score(sentence, k=self.topk)
        lang_to_score = {Lang(lang).pt1: score for lang, score in results}
        return lang_to_score.get(lan, self.penalty_value)

    def score(self, pairs: List[Tuple[str, str]]) -> Iterator[List[float]]:
        for pair in pairs:
            yield [self.raw_score(sent, self.languages[idx]) for idx, sent in enumerate(pair)]

    def accept(self, score: Tuple[float, float]) -> bool:
        return all(conf < threshold for conf, threshold in zip(score, self.thresholds))


class HeliportProbabilityFilter(HeliportRawScoreFilter):
    """Language identification filter based on normalized HeLI-OTS scores

    See :cite:`jauhiainen-etal-2022-heli` and https://github.com/ZJaume/heliport

    """

    score_direction = CLEAN_HIGH
    penalty_value = 7
    accept_threshold = -1e-9
    reject_threshold = 1

    def probability(self, sentence: str, lan: str) -> float:
        """Return probability of the sentence belonging to the language"""
        if not sentence:
            # Prevent filtering empty lines
            return 1.0
        results = self.identifier.identify_topk_with_score(sentence, k=self.topk)
        # Filter out penalty values (7)
        results = [(lang, score) for lang, score in results if score < self.penalty_value]
        values = np.array([score for lang, score in results])
        probs = 10**-(values - values.min())  # cost differences to probability-like values
        probs = probs / probs.sum()           # normalize to probabilities
        lang_to_p = {Lang(lang).pt1: probs[idx].item() for idx, (lang, score) in enumerate(results)}
        return lang_to_p.get(lan, 0.0)

    def score(self, pairs: List[Tuple[str, str]]) -> Iterator[List[float]]:
        for pair in pairs:
            yield [self.probability(sent, self.languages[idx]) for idx, sent in enumerate(pair)]

    def accept(self, score: Tuple[float, float]) -> bool:
        return all(conf > threshold for conf, threshold in zip(score, self.thresholds))
