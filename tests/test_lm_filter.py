
import argparse
import logging
import os
import tempfile
import unittest

from opusfilter import lm, subwords, OpusFilterRuntimeError, ConfigurationError


try:
    import varikn
except ImportError:
    logging.warning("Could not load varikn, language model filtering not supported")



class TestLMTokenizer(unittest.TestCase):

    traindata = ['koira jahtasi kissaa',
                 'kissa kiipesi puuhun',
                 'puu huojui tuulessa',
                 'koira haukkui maassa ja kissa sähisi puussa',
                 'melu peittyi tuuleen']

    def _train_bpe_model(self, modelfile):
        with tempfile.NamedTemporaryFile('w+') as datafile:
            for line in self.traindata:
                datafile.write(line + '\n')
            datafile.seek(0)
            subwords.BPESegmentation.train(datafile.name, modelfile)

    def _train_morfessor_model(self, modelfile):
        with tempfile.NamedTemporaryFile('w+') as datafile:
            for line in self.traindata:
                datafile.write(line + '\n')
            datafile.seek(0)
            subwords.MorfessorSegmentation.train(datafile.name, modelfile)

    def test_bad_type(self):
        with self.assertRaises(ConfigurationError):
            tokenizer = lm.LMTokenizer({'type': 'test'})

    def test_char_default(self):
        tokenizer = lm.LMTokenizer()
        tokens = tokenizer.tokenize('Hello, world! ')
        self.assertSequenceEqual(
            tokens, ['<s>', '<w>', 'H', 'e', 'l', 'l', 'o', ',', '<w>', 'w', 'o', 'r', 'l', 'd', '!', '<w>', '</s>'])

    def test_char_mb_postfix(self):
        tokenizer = lm.LMTokenizer({'type': 'char'}, mb='#$', wb='')
        tokens = tokenizer.tokenize('Hi there! ')
        self.assertSequenceEqual(
            tokens, ['<s>', 'H#', 'i', 't#', 'h#', 'e#', 'r#', 'e#', '!', '</s>'])

    def test_char_mb_prefix(self):
        tokenizer = lm.LMTokenizer({'type': 'char'}, mb='^#', wb='')
        tokens = tokenizer.tokenize('Hi there! ')
        self.assertSequenceEqual(
            tokens, ['<s>', 'H', '#i', 't', '#h', '#e', '#r', '#e', '#!', '</s>'])

    def test_bpe_mb_postfix(self):
        with tempfile.NamedTemporaryFile('w+') as modelfile:
            self._train_bpe_model(modelfile.name)
            tokenizer = lm.LMTokenizer({'type': 'bpe', 'model': modelfile.name}, mb='@@$', wb='')
            tokens = tokenizer.tokenize('koira puussa')
            self.assertSequenceEqual(
                tokens, ['<s>', 'koira', 'puu@@', 'ssa', '</s>'])

    def test_morfessor_mb_prefix(self):
        with tempfile.NamedTemporaryFile('w+') as modelfile:
            self._train_morfessor_model(modelfile.name)
            tokenizer = lm.LMTokenizer({'type': 'morfessor', 'model': modelfile.name}, mb='^#', wb='')
            tokens = tokenizer.tokenize('koira puussa')
            self.assertSequenceEqual(
                tokens, ['<s>', 'koira', 'puu', '#ssa', '</s>'])

    def test_words(self):
        tokenizer = lm.LMTokenizer({'type': 'none'}, mb='', wb='')
        tokens = tokenizer.tokenize('Hello, world! ')
        self.assertSequenceEqual(
            tokens, ['<s>', 'Hello,', 'world!', '</s>'])


@unittest.skipIf('varikn' not in globals(), 'varikn package not installed')
class TestLMTrain(unittest.TestCase):

    def test_train(self):
        with tempfile.NamedTemporaryFile('w+') as lmdatafile, tempfile.NamedTemporaryFile('w+') as lmfile:
            for line in range(10):
                lmdatafile.write('<s> <w> %s</s>\n' % ('a b <w> ' * (line + 1)))
            lmdatafile.seek(0)
            lm.train(lmdatafile.name, lmfile.name)
            self.assertGreater(len(lmfile.readlines()), 40)

    def test_train_empty(self):
        with self.assertRaises(OpusFilterRuntimeError):
            with tempfile.NamedTemporaryFile('w+') as lmdatafile, tempfile.NamedTemporaryFile('w+') as lmfile:
                lm.train(lmdatafile.name, lmfile.name)


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
