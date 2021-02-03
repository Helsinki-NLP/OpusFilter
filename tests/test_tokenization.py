import argparse
import logging
import os
import tempfile
import unittest

from opusfilter import tokenization


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
