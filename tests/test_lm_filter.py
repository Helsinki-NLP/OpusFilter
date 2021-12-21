
import argparse
import logging
import os
import tempfile
import unittest

from opusfilter import lm

try:
    import varikn
except ImportError:
    logging.warning("Could not load varikn, language model filtering not supported")


@unittest.skipIf('varikn' not in globals(), 'varikn package not installed')
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

    def test_filter_entropy_low(self):
        src_lm_params = {'filename': self.lmfile1}
        tgt_lm_params = {'filename': self.lmfile2}
        cefilter = lm.CrossEntropyFilter(
            score_type='entropy',
            thresholds=[10, 10], low_thresholds=[2, 2], diff_threshold=5,
            lm_params=[src_lm_params, tgt_lm_params])
        inputs = [('ab', 'AB'), ('abbb abbb', 'AB'), ('ab', 'BAA'), ('abbb', 'BA'), ('abbb', 'AB')]
        scores = []
        bools = []
        for score in cefilter.score(inputs):
            scores.append(score)
            bools.append(cefilter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [False, False, False, True, False])

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

    def test_filter_empty_default(self):
        src_lm_params = {'filename': self.lmfile1}
        tgt_lm_params = {'filename': self.lmfile2}
        cefilter = lm.CrossEntropyFilter(
            score_type='entropy',
            thresholds=[3, 3], diff_threshold=5,
            lm_params=[src_lm_params, tgt_lm_params])
        inputs = [('ab', 'AB'), ('', '')]
        scores = []
        bools = []
        for score in cefilter.score(inputs):
            scores.append(score)
            bools.append(cefilter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True, False])

    def test_filter_empty_pass(self):
        src_lm_params = {'filename': self.lmfile1}
        tgt_lm_params = {'filename': self.lmfile2}
        cefilter = lm.CrossEntropyFilter(
            score_type='entropy',
            thresholds=[3, 3], diff_threshold=5, score_for_empty=0,
            lm_params=[src_lm_params, tgt_lm_params])
        inputs = [('ab', 'AB'), ('', '')]
        scores = []
        bools = []
        for score in cefilter.score(inputs):
            scores.append(score)
            bools.append(cefilter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True, True])

    def test_filter_entropy_difference(self):
        id_lm_params = [{'filename': self.lmfile1}]
        nd_lm_params = [{'filename': self.lmfile2}]
        cefilter = lm.CrossEntropyDifferenceFilter(
            id_lm_params=id_lm_params, nd_lm_params=nd_lm_params,
            thresholds=[0, 0], pass_empty=False)
        inputs = [('ab',), ('ab ab',), ('Ab ab',), ('Ab Ab',), ('aB Ab',), ('AB',), ('',)]
        scores = []
        bools = []
        for score in cefilter.score(inputs):
            scores.append(score)
            bools.append(cefilter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True, True, True, False, False, False, False])

    def test_filter_entropy_difference_pass_empty(self):
        id_lm_params = [{'filename': self.lmfile1}]
        nd_lm_params = [{'filename': self.lmfile2}]
        cefilter = lm.CrossEntropyDifferenceFilter(
            id_lm_params=id_lm_params, nd_lm_params=nd_lm_params,
            thresholds=[0, 0], score_for_empty=-1)
        inputs = [('ab',), ('AB',), ('',)]
        scores = []
        bools = []
        for score in cefilter.score(inputs):
            scores.append(score)
            bools.append(cefilter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True, False, True])

    def test_filter_lm_classifier(self):
        lm_1_params = {'filename': self.lmfile1, 'interpolate': [(self.lmfile2, 0.001)]}
        lm_2_params = {'filename': self.lmfile2, 'interpolate': [(self.lmfile1, 0.001)]}
        lmfilter = lm.LMClassifierFilter(
            labels=['1', '2'], lm_params={'1': lm_1_params, '2': lm_2_params}, thresholds=[0.5, 0.5])
        inputs = [('ab', 'AB'), ('abbb abbb', 'AB'), ('ab', 'BAA'), ('abbb', 'BA'), ('abbb', 'AB'), ('abBB', 'AB'), ('ab', 'ABab')]
        scores = []
        bools = []
        for score in lmfilter.score(inputs):
            scores.append(score)
            bools.append(lmfilter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True, True, True, True, True, False, False])

    def test_filter_lm_classifier_relative(self):
        lm_1_params = {'filename': self.lmfile1, 'interpolate': [(self.lmfile2, 0.001)]}
        lm_2_params = {'filename': self.lmfile2, 'interpolate': [(self.lmfile1, 0.001)]}
        lmfilter = lm.LMClassifierFilter(
            labels=['1', '2'], lm_params={'1': lm_1_params, '2': lm_2_params}, thresholds=[1.0, 1.0], relative_score=True)
        inputs = [('ab', 'AB'), ('abbb abbb', 'AB'), ('ab', 'BAA'), ('abbb', 'BA'), ('abbb', 'AB'), ('abBB', 'AB'), ('ab', 'ABab')]
        scores = []
        bools = []
        for score in lmfilter.score(inputs):
            scores.append(score)
            bools.append(lmfilter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True, True, True, True, True, False, False])
