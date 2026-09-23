"""Filters based on the Web Docs Scorer metric

The Web Docs Scorer (https://github.com/pablop16n/web-docs-scorer) is a
tool for scoring web documents by their quality, based on properties
such as the language, punctuation, URLs, and informativeness. It is
distributed under the GPL-3.0 license and must be installed as an
optional dependency.

The scorer operates on documents that consist of one or more segments
separated by newlines. In OpusFilter, each segment of a sentence pair
is treated as a document: the segment may contain embedded newlines
(e.g. when the input is in the JSONL format) and its newline-separated
lines are scored as the segments of the document.

The filter requires a reference language code in the format
`<ISO 639-3>_<ISO 15924>` (e.g. `spa_latn`) for each segment of the
input pairs, and language identification is performed internally for
each line of the segment to provide the segment-level language codes
that the Web Docs Scorer needs.

"""

import logging
import os
import re

from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue

from . import CLEAN_HIGH, ConfigurationError, FilterABC
from .util import check_args_compability

logger = logging.getLogger(__name__)


LANG_SCRIPT_RE = re.compile(r'^[a-z]{3}_[a-z]{4}$')

# Subscore names and their indices in the detailed score output of the
# Web Docs Scorer (see score_document docstring in docscorer.py).
SUBSCORES = {
    'language_score': 1,
    'url_score': 2,
    'punctuation_score': 3,
    'singular_chars_score': 4,
    'numbers_score': 5,
    'repeated_score': 6,
    'n_long_segments_score': 7,
    'great_segment_score': 8,
    'informativeness_score': 9,
    'short_segments_score': 10,
}


def _code_to_iso639_3(code):
    """Return ISO 639-3 code for the given language code, or None

    Accepts both ISO 639-1 and ISO 639-3 codes and returns the
    corresponding ISO 639-3 code. Returns None if the code cannot be
    mapped.

    """
    try:
        return Lang(code).pt3
    except InvalidLanguageValue:
        return None


class WDSFilter(FilterABC):
    """Filtering based on the Web Docs Scorer

    Scores each segment of the input pairs as a document using the
    Web Docs Scorer. The score is the overall Web Docs Scorer score by
    default, but any of the individual subscores can be selected with
    the `subscore` parameter.

    `ref_language` defines the expected `<language>_<script>` code for
    each segment in the pairs (e.g. `['spa_latn']` for monolingual or
    `['fin_latn', 'spa_latn']` for bilingual pairs). The `threshold`
    gives the score threshold that each segment score must exceed;
    either a scalar value applied to all segments or a list with one
    value per segment.

    Language identification for the segment-level language codes is
    performed internally with the method given by `lid_method`
    (`fasttext` or `lingua`). The method-specific options are passed in
    `lid_options`. The fasttext method is expected to be used with a
    model that outputs labels in the `<language>_<script>` format (e.g.
    the OpenLID-v2 model for HPLT:
    https://huggingface.co/laurievb/OpenLID-v2). If a label consists of
    only a language code, the script is looked up from the Web Docs
    Scorer language family data, and if the language has several
    possible scripts, the script in the `ref_language` code is used.

    For the description of the Web Docs Scorer, see the package
    documentation at https://github.com/pablop16n/web-docs-scorer.

    """

    score_direction = CLEAN_HIGH
    accept_threshold = 0
    reject_threshold = 1 + 1e-6

    def __init__(
        self, ref_language=None, threshold=0, subscore=None, lid_method=None, lid_options=None, score_for_empty=1.0, **kwargs
    ):
        try:
            from docscorer.docscorer import DocumentScorer
        except ImportError:
            logger.warning("Could not load docscorer, Web Docs Scorer filtering not supported")
            raise
        super().__init__(**kwargs)
        if ref_language is None:
            raise ConfigurationError("A list of reference language codes needs to be defined")
        for code in ref_language:
            if not LANG_SCRIPT_RE.match(code):
                raise ConfigurationError(
                    f"Invalid reference language code '{code}': expected an ISO 639-3 "
                    "language code and an ISO 15924 script code separated by an "
                    "underscore (e.g. 'spa_latn')"
                )
        self.ref_language = ref_language
        self.thresholds = check_args_compability(threshold, required_types=[(int, float)], names=['threshold'])
        if isinstance(self.thresholds, list) and len(self.thresholds) != len(self.ref_language):
            raise ConfigurationError(
                f"The number of thresholds does not match the number of reference "
                f"languages: {self.thresholds} {self.ref_language}"
            )
        if subscore is not None and subscore not in SUBSCORES:
            raise ConfigurationError(f"subscore must be one of {list(SUBSCORES)}, got {subscore}")
        self.subscore = subscore
        self.score_for_empty = score_for_empty
        if lid_method not in ('fasttext', 'lingua'):
            raise ConfigurationError(f"lid_method '{lid_method}' is not supported, use 'fasttext' or 'lingua'")
        self.lid_method = lid_method
        self.lid_options = lid_options if lid_options else {}
        self.scorer = DocumentScorer()
        self._lang_scripts = self.scorer.config.df_families.groupby('language_3_chars')['script'].agg(set).to_dict()
        if lid_method == 'fasttext':
            self._init_fasttext()
        else:
            self._init_lingua()

    def _init_fasttext(self):
        """Initialize the fasttext language identifier"""
        model_path = self.lid_options.get('model_path')
        if not model_path:
            raise ConfigurationError(
                "lid_options requires a model_path pointing to a fasttext model " "(e.g. the OpenLID-v2 model)"
            )
        unknown = set(self.lid_options) - {'model_path'}
        if unknown:
            raise ConfigurationError(f"Unrecognized lid_options for fasttext: {sorted(unknown)}")
        try:
            import fasttext
        except ImportError:
            logger.warning("Could not import fasttext")
            raise
        self.fasttext_model = fasttext.load_model(os.path.join(self.workdir, model_path))

    def _init_lingua(self):
        """Initialize the lingua language identifier"""
        unknown = set(self.lid_options) - {'lingua_mode', 'langid_languages'}
        if unknown:
            raise ConfigurationError(f"Unrecognized lid_options for lingua: {sorted(unknown)}")
        from lingua import IsoCode639_1, LanguageDetectorBuilder

        langid_languages = self.lid_options.get('langid_languages')
        if langid_languages:
            for code in langid_languages:
                if not hasattr(IsoCode639_1, code.upper()):
                    raise ConfigurationError(f"Language {code} not supported by lingua")
            from_languages = LanguageDetectorBuilder.from_iso_codes_639_1(
                *[getattr(IsoCode639_1, code.upper()) for code in langid_languages]
            )
        else:
            from_languages = LanguageDetectorBuilder.from_all_languages()
        lingua_mode = self.lid_options.get('lingua_mode', 'low')
        if lingua_mode == 'high':
            self.lingua_detector = from_languages.with_preloaded_language_models().build()
        elif lingua_mode == 'low':
            self.lingua_detector = from_languages.with_low_accuracy_mode().build()
        else:
            raise ConfigurationError(f"lingua mode '{lingua_mode}' is not supported.")

    def _script_for_language(self, code, ref_code):
        """Return script code for a language code

        The language code is converted to ISO 639-3 and the script is
        looked up from the Web Docs Scorer language family data. If the
        language is not found or has several possible scripts, the
        script in the reference language code is used.

        """
        iso639_3 = _code_to_iso639_3(code)
        if iso639_3 is None:
            return ref_code.split('_')[1]
        scripts = self._lang_scripts.get(iso639_3)
        if scripts and len(scripts) == 1:
            return next(iter(scripts))
        return ref_code.split('_')[1]

    def _fasttext_lang_script(self, text, ref_code):
        """Predict language and script code with fasttext"""
        labels, _ = self.fasttext_model.predict([text], k=1)
        label = labels[0][0][9:].lower()
        if LANG_SCRIPT_RE.match(label):
            return label
        code = _code_to_iso639_3(label) or ref_code.split('_')[0]
        return f"{code}_{self._script_for_language(label, ref_code)}"

    def _lingua_lang_script(self, text, ref_code):
        """Predict language and script code with lingua"""
        confidence_values = self.lingua_detector.compute_language_confidence_values(text)
        lang = confidence_values[0].language.iso_code_639_1.name.lower()
        code = _code_to_iso639_3(lang) or ref_code.split('_')[0]
        return f"{code}_{self._script_for_language(lang, ref_code)}"

    def _lang_script(self, text, ref_code):
        """Return the predicted '*_#' language code for a segment"""
        if not text.strip():
            # Prevent language identification on empty lines
            return ref_code
        if self.lid_method == 'fasttext':
            return self._fasttext_lang_script(text, ref_code)
        return self._lingua_lang_script(text, ref_code)

    def _score_document(self, document, ref_code):
        """Return the score for a single document"""
        if not document.strip():
            # Prevent scoring empty documents
            return self.score_for_empty
        ref_lang, ref_script = ref_code.split('_')
        lang_segments = [self._lang_script(segment, ref_code) for segment in document.split('\n')]
        result = self.scorer.score_document(
            ref_lang=ref_lang,
            ref_script=ref_script,
            lang_segments=lang_segments,
            document_text=document,
            doc_id='',
            raw_score=self.subscore is None,
        )
        if self.subscore is None:
            return float(result)
        if len(result) <= SUBSCORES[self.subscore]:
            raise RuntimeError(f"Too few subscores in docscorer output: {result}")
        return float(result[SUBSCORES[self.subscore]])

    def score(self, pairs):
        for pair in pairs:
            if len(pair) != len(self.ref_language):
                raise ValueError(
                    f"Input pair length {len(pair)} does not match the number of " f"reference languages: {self.ref_language}"
                )
            yield [self._score_document(document, ref_code) for document, ref_code in zip(pair, self.ref_language)]

    def accept(self, score):
        return all(value > threshold for value, threshold in zip(score, self.thresholds))
