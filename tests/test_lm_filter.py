
import argparse
import logging
import os
import tempfile
import unittest

from opusfilter import lm


# TODO: tests for LMTokenizer


class TestLMFilter(unittest.TestCase):

    def setUp(self):
        self.lmdatafile1 = tempfile.mkstemp()[1]
        self.lmfile1 = tempfile.mkstemp()[1]
        with open(self.lmdatafile1, 'w') as lmdatafile:
            for line in range(10):
                lmdatafile.write('<s> <w> %s</s>\n' % ('a b <w> ' * (line + 1)))
        self.lmdatafile2 = tempfile.mkstemp()[1]
        self.lmfile2 = tempfile.mkstemp()[1]
        with open(self.lmdatafile2, 'w') as lmdatafile:
            for line in range(10):
                lmdatafile.write('<s> <w> %s</s>\n' % ('A B <w> ' * (line + 1)))
        lm.train(self.lmdatafile1, self.lmfile1)
        lm.train(self.lmdatafile2, self.lmfile2)
        logging.info(self.lmfile1)
        with open(self.lmfile1, 'r') as fobj:
            for line in fobj:
                logging.info(line.strip())
        logging.info(self.lmfile2)
        with open(self.lmfile2, 'r') as fobj:
            for line in fobj:
                logging.info(line.strip())

    def tearDown(self):
        os.remove(self.lmdatafile1)
        os.remove(self.lmfile1)
        os.remove(self.lmdatafile2)
        os.remove(self.lmfile2)

    def test_filter_entropy(self):
        src_lm_params = {'filename': self.lmfile1}
        tgt_lm_params = {'filename': self.lmfile2}
        cefilter = lm.CrossEntropyFilter(
            score_type='entropy',
            thresholds=[10, 10], diff_threshold=5,
            lm_params=[src_lm_params, tgt_lm_params])
        inputs = [('ab', 'AB'), ('abbb abbb', 'AB'), ('ab', 'BAA'), ('abbb', 'BA'), ('abbb', 'AB')]
        scores = []
        bools = []
        for score in cefilter.score(inputs):
            scores.append(score)
            bools.append(cefilter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True, False, False, True, False])

    def test_filter_perplexity(self):
        src_lm_params = {'filename': self.lmfile1}
        tgt_lm_params = {'filename': self.lmfile2}
        cefilter = lm.CrossEntropyFilter(
            score_type='perplexity',
            thresholds=[1000, 1000], diff_threshold=100,
            lm_params=[src_lm_params, tgt_lm_params])
        inputs = [('ab', 'AB'), ('abbb abbb', 'AB'), ('ab', 'BAA'), ('abbb', 'BA'), ('abbb', 'AB')]
        scores = []
        bools = []
        for score in cefilter.score(inputs):
            scores.append(score)
            bools.append(cefilter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True, False, False, False, False])

    def test_filter_logprob(self):
        src_lm_params = {'filename': self.lmfile1}
        tgt_lm_params = {'filename': self.lmfile2}
        cefilter = lm.CrossEntropyFilter(
            score_type='logprob',
            thresholds=[20, 20], diff_threshold=5,
            lm_params=[src_lm_params, tgt_lm_params])
        inputs = [('ab', 'AB'), ('abbb abbb', 'AB'), ('ab', 'BAA'), ('abbb', 'BA'), ('abbb', 'AB')]
        scores = []
        bools = []
        for score in cefilter.score(inputs):
            scores.append(score)
            bools.append(cefilter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True, False, False, True, False])
