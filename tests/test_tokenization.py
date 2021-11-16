import logging
import unittest

from opusfilter import tokenization

try:
    import jieba
except ImportError:
    logging.warning("Could not import jieba")


class TestTokenization(unittest.TestCase):

    def test_dummy(self):
        tokenize = tokenization.get_tokenize(None)
        self.assertEqual(tokenize("Hello, world!"), "Hello, world!")

    def test_dummy_detok(self):
        tokenize = tokenization.get_tokenize(None)
        self.assertEqual(tokenize.detokenize("Hello , world !"), "Hello , world !")

    def test_moses(self):
        tokenize = tokenization.get_tokenize(('moses', 'en'))
        self.assertEqual(tokenize("Hello, world!"), "Hello , world !")

    def test_moses_detok(self):
        tokenize = tokenization.get_tokenize(('moses', 'en'))
        self.assertEqual(tokenize.detokenize("Hello , world !"), "Hello, world!")

    def test_moses_options(self):
        tokenize = tokenization.get_tokenize(('moses', 'en', {'aggressive_dash_splits': True}))
        self.assertEqual(tokenize("Hello, fine-looking world!"), "Hello , fine @-@ looking world !")

    @unittest.skipIf('jieba' not in globals(), 'jieba not installed')
    def test_jieba(self):
        tokenize = tokenization.get_tokenize(('jieba', 'en'))
        text = "同时，祖马革命的一代似乎对领导打破种族隔离制度15年后的南非，还不适应。"
        # The expected word segmentation result is not directly given here, because there is no absolute
        # standard word segmentation result in Chinese.
        # Different versions of the model will output slightly different word segmentation results.
        # So just assert the max length of generated tokens is less than a small value (just use 8 here).
        token_max_len = max(tokenize(text).split(), key=lambda x: len(x))
        self.assertLess(len(token_max_len), 8)

    @unittest.skipIf('jieba' not in globals(), 'jieba not installed')
    def test_jieba_detok(self):
        tokenize = tokenization.get_tokenize(('jieba', 'en'))
        tokens = "同时 ， 祖马 革命 的 一代 似乎 对 领导 打破 种族隔离 制度 15 年 后 的 南非 ， 还 不 适应 。"
        reference = "同时，祖马革命的一代似乎对领导打破种族隔离制度15年后的南非，还不适应。"
        self.assertEqual(tokenize.detokenize(tokens), reference)
