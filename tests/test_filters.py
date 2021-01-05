import copy
import unittest

from opusfilter.filters import *


class TestLongestCommonSubstringFilter(unittest.TestCase):

    bi_inputs = [('abcd', 'abcd'), ('abcd', 'efgh'), ('abcd', 'cdgh'), ('abcd', ''), ('', ''),
                 ('abcd', 'bc'), ('abcd', 'ab abcd cd'), ('abcd ', ' abcd'), ('ab cd', 'a bc d')]
    tri_inputs = [('abcd', 'abcd', 'abcd'), ('abcd', 'abcd', 'efgh'), ('abcd', '', ''), ('', '', ''),
                  ('abcd', 'abc', 'bc'), ('abcd', 'xbcd', 'xabx')]
    
    def test_bilingual(self):
        testfilter = LongestCommonSubstringFilter(threshold=0.8, require_all=True)
        expected = [([1], False), ([0], True), ([0.5], True), ([0], True), ([0], True),
                    ([1], False), ([1], False), ([0.8], False), ([0.2], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.bi_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_trilingual(self):
        testfilter = LongestCommonSubstringFilter(threshold=0.75, require_all=True)
        expected = [([1, 1, 1], False), ([1, 0, 0], False), ([0, 0, 0], True), ([0, 0, 0], True),
                    ([1, 1, 1], False), ([0.75, 0.5, 0.25], False)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.tri_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

    def test_trilingual_any(self):
        testfilter = LongestCommonSubstringFilter(threshold=0.75, require_all=False)
        expected = [([1, 1, 1], False), ([1, 0, 0], True), ([0, 0, 0], True), ([0, 0, 0], True),
                    ([1, 1, 1], False), ([0.75, 0.5, 0.25], True)]
        results = [(x, testfilter.accept(x)) for x in testfilter.score(self.tri_inputs)]
        for result, correct in zip(results, expected):
            self.assertSequenceEqual(result, correct)

