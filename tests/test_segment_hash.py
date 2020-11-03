import itertools
import logging
import unittest

from opusfilter.segment_hash import SegmentHasher


class TestFilterPipeline(unittest.TestCase):

    segment = "aa " * 10
    segment_different_char = "bb " * 10
    segment_different_len = "aa " * 11
    segment_duplicate = "aa " * 10
    segment_uppercase = "AA " * 10
    segment_extra_whitespace = " aa " * 10
    segment_extra_punct = "aa, " * 10
    segment_uppercase_extra_whitespace = " AA " * 10
    all_variants = [
        segment_different_char, segment_different_len, segment_duplicate,
        segment_uppercase, segment_extra_whitespace, segment_extra_punct,
        segment_uppercase_extra_whitespace
    ]

    def test_letters_only_preprocess(self):
        """Test preprocessing options"""
        letters = ['A', 'Æ', 'ƺ', 'ɸ', 'ʷ', 'Ѯ', 'ն', 'ע', 'ࠕ', 'ছ', 'ผ',
                   'ወ', 'ᑔ', 'ヨ', '㓦']
        non_letters = [' ', '\n', '\r', '\t', '-', '_', '!', '$', '&', '*',
                       '(', '5', '§', '˧', '׃', '‰', '∫']
        hasher = SegmentHasher(letters_only=True)
        for example in letters:
            logging.debug("Testing letter %s", example)
            self.assertEqual(hasher.preprocess(example), example)
        for example in non_letters:
            logging.debug("Testing non-letter %s", example)
            self.assertEqual(hasher.preprocess(example), '')

    def _test_multiple(self, hasher, segment, variants, results):
        for variant, result in zip(variants, results):
            match = (hasher.apply(segment) == hasher.apply(variant))
            logging.debug("Testing %s %s %s", segment, '==' if result else '!=', variant)
            self.assertEqual(match, result)

    def test_single_default(self):
        """Single source, default options"""
        self._test_multiple(
            SegmentHasher(),
            [self.segment],
            [[x] for x in self.all_variants],
            [False, False, True, False, False, False, False]
        )

    def test_single_lowercase(self):
        """Single source, default options"""
        self._test_multiple(
            SegmentHasher(lowercase=True),
            [self.segment],
            [[x] for x in self.all_variants],
            [False, False, True, True, False, False, False]
        )

    def test_single_letters(self):
        """Single source, default options"""
        self._test_multiple(
            SegmentHasher(letters_only=True),
            [self.segment],
            [[x] for x in self.all_variants],
            [False, False, True, False, True, True, False]
        )

    def test_single_letters_lowercase(self):
        """Single source, default options"""
        self._test_multiple(
            SegmentHasher(lowercase=True, letters_only=True),
            [self.segment],
            [[x] for x in self.all_variants],
            [False, False, True, True, True, True, True]
        )

    def test_pairs(self):
        """Single source, default options"""
        # 2nd segments always same, compare all
        self._test_multiple(
            SegmentHasher(compare='all'),
            [self.segment, 'bb ' * 10],
            [[x, 'bb ' * 10] for x in self.all_variants],
            [False, False, True, False, False, False, False]
        )
        # 2nd segments always same, ignore 1st segments
        self._test_multiple(
            SegmentHasher(compare=[1]),
            [self.segment, 'bb ' * 10],
            [[x, 'bb ' * 10] for x in self.all_variants],
            [True, True, True, True, True, True, True, True]
        )
        # 2nd segments always different, compare all
        self._test_multiple(
            SegmentHasher(compare='all'),
            [self.segment, 'bb ' * 10],
            [[x, 'cc ' * 10] for x in self.all_variants],
            [False, False, False, False, False, False, False]
        )
        # 2nd segments always different, ignore 2nd segments
        self._test_multiple(
            SegmentHasher(compare=[0]),
            [self.segment, 'bb ' * 10],
            [[x, 'cc ' * 10] for x in self.all_variants],
            [False, False, True, False, False, False, False]
        )

    def test_pairs_join(self):
        """Test that segment joining cannot be fooled"""
        hasher = SegmentHasher()
        for seg1, seg2 in [
                (('aaa', 'bbb'), ('', 'aaabbb')),
                (('aaa', 'bbb'), ('aaab', 'bb')),
                (('aaa', 'bbb'), ('aaabbb', '')),
                ((hasher.join_char + 'aaa', 'bbb'), ('', 'aaa' + hasher.join_char + 'bbb')),
                (('aaa', 'bbb' + hasher.join_char), ('aaa' + hasher.join_char + 'bbb', '')),
        ]:
            out1, out2 = hasher.apply(seg1), hasher.apply(seg2)
            self.assertNotEqual(out1, out2)
