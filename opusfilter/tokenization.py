"""Tokenization tools"""

import logging

from . import ConfigurationError


logger = logging.getLogger(__name__)


try:
    import mosestokenizer
except ImportError:
    logger.warning("Could not import mosestokenizer, moses tokenization not supported")


class DummyTokenizer:
    """Dummy tokenizer"""

    def tokenize(self, string):
        """Return tokenized version of the input string"""
        return string

    def __call__(self, string):
        """Return tokenized version of the input string"""
        return self.tokenize(string)


class MosesTokenizer(DummyTokenizer):
    """Wrapper for mosestokenizer.MosesTokenizer"""

    def __init__(self, lang):
        try:
            self._moses_tokenizer = mosestokenizer.MosesTokenizer(lang)
        except NameError as err:
            logger.error("Install mosestokenizer to support moses tokenization")
            raise err

    def tokenize(self, string):
        return ' '.join(self._moses_tokenizer(string))


def get_tokenize(specs):
    """Return object that returns a tokenized version of the input string on call"""
    if specs is None:
        return DummyTokenizer()
    if not (isinstance(specs, (list, tuple)) and len(specs) == 2):
        raise ConfigurationError(
            "Tokenizer definition should be None or a (type, language code) pair")
    tokenizer, lang = specs
    if tokenizer == 'moses':
        return MosesTokenizer(lang)
    else:
        raise ConfigurationError("Tokenizer type '%s' not supported" % tokenizer)
