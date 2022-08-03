import logging
import unittest

from opusfilter import tokenization, ConfigurationError

try:
    import jieba
except ImportError:
    logging.warning("Could not import jieba")

try:
    import MeCab
except ImportError:
    logging.warning("Could not import MeCab")


class TestTokenization(unittest.TestCase):

    def test_unknown(self):
        with self.assertRaises(ConfigurationError):
            tokenize = tokenization.get_tokenize(('the_best_tokenizer'))

    def test_dummy(self):
        tokenize = tokenization.get_tokenize(None)
        self.assertEqual(tokenize("Hello, world!"), "Hello, world!")

    def test_dummy_detok(self):
        tokenize = tokenization.get_tokenize(None)
        self.assertEqual(tokenize.detokenize("Hello , world !"), "Hello , world !")

    def test_moses(self):
        tokenize = tokenization.get_tokenize(('moses', 'en'))
        self.assertEqual(tokenize("Hello, world!"), "Hello , world !")

    def test_moses_fallback(self):
        with self.assertLogs() as captured:
            tokenize = tokenization.get_tokenize(('moses', 'xx'))
        self.assertIn("fall-back to English", captured.records[0].getMessage())
        self.assertEqual(tokenize("Hello, world!"), "Hello , world !")

    def test_moses_detok(self):
        tokenize = tokenization.get_tokenize(('moses', 'en'))
        self.assertEqual(tokenize.detokenize("Hello , world !"), "Hello, world!")

    def test_moses_options(self):
        tokenize = tokenization.get_tokenize(('moses', 'en', {'aggressive_dash_splits': True}))
        self.assertEqual(tokenize("Hello, fine-looking world!"), "Hello , fine @-@ looking world !")

    @unittest.skipIf('jieba' not in globals(), 'jieba not installed')
    def test_jieba(self):
        tokenize = tokenization.get_tokenize(('jieba', 'zh'))
        text = "同时，祖马革命的一代似乎对领导打破种族隔离制度15年后的南非，还不适应。"
        # The expected word segmentation result is not directly given here, because there is no absolute
        # standard word segmentation result in Chinese.
        # Different versions of the model will output slightly different word segmentation results.
        # So just assert the max length of generated tokens is less than a small value (just use 8 here).
        token_max_len = max(tokenize(text).split(), key=lambda x: len(x))
        self.assertLess(len(token_max_len), 8)

    @unittest.skipIf('jieba' not in globals(), 'jieba not installed')
    def test_jieba_detok(self):
        tokenize = tokenization.get_tokenize(('jieba', 'zh', {'map_space_to': '␣'}))
        tokens = "同时 ， 祖马 革命 的 一代 似乎 对 领导 打破 种族隔离 制度 15 年 后 的 南非 （ South ␣ Africa ) ， 还 不 适应 。"
        reference = "同时，祖马革命的一代似乎对领导打破种族隔离制度15年后的南非（South Africa)，还不适应。"
        self.assertEqual(tokenize.detokenize(tokens), reference)

    @unittest.skipIf('jieba' not in globals(), 'jieba not installed')
    def test_jieba_tok_and_detok(self):
        tokenize = tokenization.get_tokenize(('jieba', 'zh', {'map_space_to': '␣'}))
        text = " 年后的南非（South Africa)，还不适应。 "
        self.assertEqual(tokenize.detokenize(tokenize(text)), text)

    @unittest.skipIf('jieba' not in globals(), 'jieba not installed')
    def test_jieba_non_zh(self):
        with self.assertLogs() as captured:
            tokenize = tokenization.get_tokenize(('jieba', 'en'))
        self.assertIn("tokenizer only avaliable for Chinese", captured.records[0].getMessage())

    @unittest.skipIf('MeCab' not in globals(), 'MeCab not installed')
    def test_mecab(self):
        tokenize = tokenization.get_tokenize(('mecab', 'jp'))
        text = "これは英語で書く必要はありません。"
        self.assertEqual(tokenize(text), "これ は 英語 で 書く 必要 は あり ませ ん 。")

    @unittest.skipIf('MeCab' not in globals(), 'MeCab not installed')
    def test_mecab_detok(self):
        tokenize = tokenization.get_tokenize(('mecab', 'jp', {'map_space_to': '␣'}))
        tokens = "これ は 英語 ␣ ( not ␣ here ) ␣ で 書く 必要 は あり ませ ん 。"
        reference = "これは英語 (not here) で書く必要はありません。"
        self.assertEqual(tokenize.detokenize(tokens), reference)

    @unittest.skipIf('MeCab' not in globals(), 'MeCab not installed')
    def test_mecab_tok_and_detok(self):
        tokenize = tokenization.get_tokenize(('mecab', 'jp', {'map_space_to': '␣'}))
        text = " これは英語 (not here) で書く必要はありません。 "
        self.assertEqual(tokenize.detokenize(tokenize(text)), text)

    @unittest.skipIf('MeCab' not in globals(), 'MeCab not installed')
    def test_mecab_non_jp(self):
        with self.assertLogs() as captured:
            tokenize = tokenization.get_tokenize(('mecab', 'en'))
        self.assertIn("tokenizer is for Japanese", captured.records[0].getMessage())
