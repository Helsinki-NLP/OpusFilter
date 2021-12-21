import logging
import unittest

from opusfilter.pipeline import PreprocessorPipeline
from opusfilter.preprocessors import *

try:
    import jieba
except ImportError:
    logging.warning("Could not import jieba")


UNICODE_WHITESPACE_CHARACTERS = [
    "\u0009",  # character tabulation
    "\u000a",  # line feed
    "\u000b",  # line tabulation
    "\u000c",  # form feed
    "\u000d",  # carriage return
    "\u0020",  # space
    "\u0085",  # next line
    "\u00a0",  # no-break space
    "\u1680",  # ogham space mark
    "\u2000",  # en quad
    "\u2001",  # em quad
    "\u2002",  # en space
    "\u2003",  # em space
    "\u2004",  # three-per-em space
    "\u2005",  # four-per-em space
    "\u2006",  # six-per-em space
    "\u2007",  # figure space
    "\u2008",  # punctuation space
    "\u2009",  # thin space
    "\u200A",  # hair space
    "\u2028",  # line separator
    "\u2029",  # paragraph separator
    "\u202f",  # narrow no-break space
    "\u205f",  # medium mathematical space
    "\u3000",  # ideographic space
]


class TestWhitespaceNormalizer(unittest.TestCase):

    processor = WhitespaceNormalizer()

    def test_trailing(self):
        inputs = ["hello", "hello ", "hello  ", "hello\t\r", "hello\u2004"]
        expected = ["hello"] * 5
        results = list(self.processor.process([x] for x in inputs))
        for result, correct in zip(results, expected):
            self.assertEqual(result, [correct])

    def test_leading(self):
        inputs = ["hello", " hello", "  hello", "\thello", "\u3000hello"]
        expected = ["hello"] * 5
        results = list(self.processor.process([x] for x in inputs))
        for result, correct in zip(results, expected):
            self.assertEqual(result, [correct])

    def test_multiple(self):
        inputs = ["hello world", "hello  world", "hello    world",
                  "hello\tworld", "hello \t  world\r"]
        expected = ["hello world"] * 5
        results = list(self.processor.process([x] for x in inputs))
        for result, correct in zip(results, expected):
            self.assertEqual(result, [correct])

    def test_special(self):
        inputs = ["hello" + char + "world" for char in UNICODE_WHITESPACE_CHARACTERS]
        expected = ["hello world"] * len(inputs)
        results = list(self.processor.process([x] for x in inputs))
        for result, correct in zip(results, expected):
            self.assertEqual(result, [correct])


class TestTokenizer(unittest.TestCase):

    def test_moses_fi_en(self):
        tokenizer = Tokenizer('moses', ['fi', 'en'])
        detokenizer = Detokenizer('moses', ['fi', 'en'])
        detokenized = list(zip(*[
            ["Hello, world!", "Punctuation, e.g., comma", "C. done 4.5"],
            ["Hei, maailma!", "Välimerkit, esim. pilkku", "C. valmis 4,5"]
        ]))
        tokenized = list(zip(*[
            ["Hello , world !", "Punctuation , e.g. , comma", "C. done 4.5"],
            ["Hei , maailma !", "Välimerkit , esim. pilkku", "C. valmis 4,5"]
        ]))
        results = list(tokenizer.process(detokenized))
        for result, expected in zip(results, tokenized):
            self.assertSequenceEqual(result, expected)
        results = list(detokenizer.process(tokenized))
        for result, expected in zip(results, detokenized):
            self.assertSequenceEqual(result, expected)

    @unittest.skipIf('jieba' not in globals(), 'jieba not installed')
    def test_two_tokenizers(self):
        tokenizer = Tokenizer(['jieba', 'moses'], ['zh', 'en'])
        detokenizer = Detokenizer(['jieba', 'moses'], ['zh', 'en'])
        detokenized = list(zip(*[
            ["你好，世界！", "你好吗?"],
            ["Hello, world!", "How are you?"],
        ]))
        tokenized = list(zip(*[
            ["你好 ， 世界 ！", "你好 吗 ?"],
            ["Hello , world !", "How are you ?"]
        ]))
        results = list(tokenizer.process(detokenized))
        for result, expected in zip(results, tokenized):
            self.assertSequenceEqual(result, expected)
        results = list(detokenizer.process(tokenized))
        for result, expected in zip(results, detokenized):
            self.assertSequenceEqual(result, expected)

    @unittest.skipIf('jieba' not in globals(), 'jieba not installed')
    def test_two_tokenizers_and_options(self):
        tokenizer = Tokenizer(['jieba', 'moses'], ['zh', 'en'], [{'cut_all': True}, {}])
        detokenizer = Detokenizer(['jieba', 'moses'], ['zh', 'en'])
        detokenized = list(zip(*[
            ["你好，世界！", "你好吗?"],
            ["Hello, world!", "How are you?"],
        ]))
        tokenized = list(zip(*[
            ["你好 ， 世界 ！", "你好 吗 ?"],
            ["Hello , world !", "How are you ?"]
        ]))
        results = list(tokenizer.process(detokenized))
        for result, expected in zip(results, tokenized):
            self.assertSequenceEqual(result, expected)
        results = list(detokenizer.process(tokenized))
        for result, expected in zip(results, detokenized):
            self.assertSequenceEqual(result, expected)


class TestRegExpSub(unittest.TestCase):

    def test_single(self):
        inputs = ["hello", "(1) hello", " (24) hello", "(4-2) hello", "hello (0)"]
        expected = ["hello"] * 4 + ["hello (0)"]
        processor = RegExpSub([(r"^ *\([0-9-]+\) *", "", 0, [])])
        results = list(processor.process([x] for x in inputs))
        for result, correct in zip(results, expected):
            self.assertEqual(result, [correct])

    def test_lang_patterns(self):
        inputs = zip(*[["hello", "(1) hello", " (24) hello", "(4-2) hello", "hello (0)"],
                       ["hei"] * 4 + ["(1) hei"]])
        outputs = zip(*[["hello"] * 4 + ["hello (0)"], ["hei"] * 4 + ["(1) hei"]])
        # Use substitution only for the first language index
        processor = RegExpSub(
            [(r"^ *\([0-9-]+\) *", "", 0, [])],
            lang_patterns={1: []}
        )
        results = list(processor.process(inputs))
        for result, expected in zip(results, outputs):
            self.assertSequenceEqual(result, expected)


class TestPreprocessorPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.config = [
            {'WhitespaceNormalizer': {}},
            {'RegExpSub': {'patterns': ([(r"^ *\([0-9-]+\) *", "", 0, [])])}},
            {'Tokenizer': {'tokenizer': 'moses', 'languages': ['fi', 'en']}},
        ]

    def test_from_config(self):
        pipe = PreprocessorPipeline.from_config(self.config)
        self.assertEqual(len(pipe.preprocessors), 3)
        inputs = zip(*[
            ["Hello, world!", "(1) Punctuation, e.g., comma", "(2) C. done 4.5"],
            ["Hei, maailma!", "(1) Välimerkit, esim. pilkku", "(2) C. valmis 4,5"]
        ])
        outputs = zip(*[
            ["Hello , world !", "Punctuation , e.g. , comma", "C. done 4.5"],
            ["Hei , maailma !", "Välimerkit , esim. pilkku", "C. valmis 4,5"]
        ])
        results = list(pipe.process(inputs))
        for result, expected in zip(results, outputs):
            self.assertSequenceEqual(result, expected)


class TestMonolingualSentenceSplitter(unittest.TestCase):

    def test_default(self):
        inputs = [["Hello! How are you? I feel great."], ["Oh, no"]]
        expected = ["Hello!", "How are you?", "I feel great.", "Oh, no"]
        processor = MonolingualSentenceSplitter(language='en')
        results = processor.process(inputs)
        for result, correct in zip(results, expected):
            self.assertEqual(result, [correct])

    def test_parallel_error(self):
        inputs = [["Hello!", "Hei"], ["How are you?", "Mitä kuuluu"]]
        processor = MonolingualSentenceSplitter(language='en')
        with self.assertRaises(ConfigurationError):
            results = list(processor.process(inputs))

    def test_parallel_enabled(self):
        inputs = [["Hello!", "Hei"], ["How are you?", "Mitä kuuluu"]]
        processor = MonolingualSentenceSplitter(language='en', enable_parallel=True)
        results = processor.process(inputs)
        for result, correct in zip(results, inputs):
            self.assertEqual(result, correct)
