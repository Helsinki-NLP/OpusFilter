"""Tokenization tools"""

import logging

from . import ConfigurationError


logger = logging.getLogger(__name__)


try:
    import mosestokenizer
except ImportError:
    logger.warning("Could not import mosestokenizer, moses tokenization not supported")

try:
    import jieba
except ImportError:
    logger.warning("Could not import jieba, jieba tokenization not supported")


class DummyTokenizer:
    """Dummy tokenizer"""

    @staticmethod
    def tokenize(string):
        """Return tokenized version of the input string"""
        return string

    @staticmethod
    def detokenize(string):
        """Return detokenized version of the input string"""
        return string

    def __call__(self, string):
        """Return tokenized version of the input string"""
        return self.tokenize(string)


class MosesTokenizer(DummyTokenizer):
    """Wrapper for mosestokenizer.MosesTokenizer

    Options for MosesTokenizer:

      aggressive_dash_splits (bool, optional):
          Aggressively split hyphens. Defaults to False.
      escape_xml (bool, optional):
          Escape XML characters. Defaults to False.
      unescape_xml (bool, optional):
          Unescape XML characters. Defaults to False.
      preserve_xml_entities (bool, optional):
          Preserve HTML entities. Defaults to False.
      refined_punct_splits (bool, optional):
          Refined punctuation splitting. Defaults to False.
      url_handling (bool, optional): Handle URLs. Defaults to True.
      supersub (bool, optional): Account for numerical super and
          subscript conjoining. Defaults to False.
      penn (bool, optional): Use PENN tokenizer. Defaults to False.
      verbose (bool, optional): Print messages. Defaults to False.
      user_dir (Optional[str], optional): User provided nonbreaking
          prefixes and protected patterns. Defaults to None.

    """

    def __init__(self, lang, **options):
        try:
            self._moses_tokenizer = mosestokenizer.MosesTokenizer(lang, **options)
        except RuntimeError as err:
            msg = str(err)
            if 'No known abbreviations for language' in msg:
                logger.warning("%s - attempting fall-back to English version", msg)
                self._moses_tokenizer = mosestokenizer.MosesTokenizer('en')
            else:
                raise err
        except NameError as err:
            logger.error("Install fast-mosestokenizer to support moses tokenization")
            raise err

    def tokenize(self, string):
        return ' '.join(self._moses_tokenizer.tokenize(string))

    def detokenize(self, string):
        return self._moses_tokenizer.detokenize(string.split())


class JiebaTokenizer(DummyTokenizer):
    """Wrapper for popular Chinese tokenizer jieba"""

    def __init__(self, lang, **options):
        if not lang.startswith('zh'):
            logger.warning("Jieba tokenizer only avaliable for Chinese (zh), requested language %s", lang)
        try:
            self.jieba = jieba
        except NameError as err:
            logger.error("Install jieba to support jieba tokenization")
            raise err
        self.options = options

    def tokenize(self, string):
        return ' '.join(self.jieba.cut(string, **self.options))

    def detokenize(self, string):
        return ''.join(string.split())


def get_tokenize(specs):
    """Return object that returns a tokenized version of the input string on call"""
    if specs is None:
        return DummyTokenizer()
    if not (isinstance(specs, (list, tuple)) and len(specs) in {2, 3}):
        raise ConfigurationError(
            "Tokenizer definition should be None or a list or tuple (type, language code[, options])")
    tokenizer = specs[0]
    lang = specs[1]
    options = specs[2] if len(specs) > 2 else None
    if options is None:
        options = {}
    if tokenizer == 'moses':
        return MosesTokenizer(lang, **options)
    if tokenizer == 'jieba':
        return JiebaTokenizer(lang, **options)
    raise ConfigurationError(f"Tokenizer type '{tokenizer}' not supported")
