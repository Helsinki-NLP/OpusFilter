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

    def detokenize(self, string):
        """Return detokenized version of the input string"""
        return string

    def __call__(self, string):
        """Return tokenized version of the input string"""
        return self.tokenize(string)


class MosesTokenizer(DummyTokenizer):
    """Wrapper for mosestokenizer.MosesTokenizer"""

    def __init__(self, lang):
        try:
            self._moses_tokenizer = mosestokenizer.MosesTokenizer(lang)
        except RuntimeError as err:
            msg = str(err)
            if 'No known abbreviations for language' in msg:
                logger.warning(msg + " - attempting fall-back to English version")
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
