
import argparse
import json
import logging
import os
import tempfile
import unittest

from opusfilter import word_alignment, OpusFilterRuntimeError


@unittest.skipIf(os.environ.get('EFLOMAL_PATH') is None, 'EFLOMAL_PATH not defined in environment')
class TestAlignFilter(unittest.TestCase):

    def test_scoring(self):
        """Test word alignment scoring on artificial data

        Note: The result of the alignment is not deterministic and the
        test could fail with really bad luck.

        """
        data1 = ['%s.' % ('ab ' * (line + 1)) for line in range(10)] * 5 + ['ab ab ab ab ab ab .', ''] * 2
        data2 = ['%s.' % ('AB ' * (line + 1)) for line in range(10)] * 5 + ['AB', ''] * 2
        logging.info(data1)
        logging.info(data2)
        align_filter = word_alignment.WordAlignFilter(src_threshold=0, tgt_threshold=0)
        scores = []
        bools = []
        for score in align_filter.score(zip(data1, data2)):
            scores.append(score)
            bools.append(align_filter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True] * 50 + [False, True, False, True])

    def test_scoring_for_empty(self):
        """Test word alignment scoring on artificial data

        Note: The result of the alignment is not deterministic and the
        test could fail with really bad luck.

        """
        data1 = ['%s.' % ('ab ' * (line + 1)) for line in range(10)] * 5 + ['ab ab ab ab ab ab .', ''] * 2
        data2 = ['%s.' % ('AB ' * (line + 1)) for line in range(10)] * 5 + ['AB', ''] * 2
        logging.info(data1)
        logging.info(data2)
        align_filter = word_alignment.WordAlignFilter(src_threshold=0, tgt_threshold=0, score_for_empty=0)
        scores = []
        bools = []
        for score in align_filter.score(zip(data1, data2)):
            scores.append(score)
            bools.append(align_filter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True] * 50 + [False, False, False, False])

    def test_make_priors(self):
        """Test making priors

        Note: The result of the alignment is not deterministic and the
        test could fail with really bad luck.

        """
        prior_data1 = ['%s.' % ('ab ' * (line + 1)) for line in range(10)] * 5
        prior_data2 = ['%s.' % ('AB ' * (line + 1)) for line in range(10)] * 5
        with tempfile.NamedTemporaryFile('w+') as priors_file:
            word_alignment.make_priors(zip(prior_data1, prior_data2), priors_file.name)
            self.assertTrue(priors_file.read())

    def test_make_priors_empty(self):
        """Test making priors with empty data"""
        with self.assertRaises(OpusFilterRuntimeError):
            with tempfile.NamedTemporaryFile('w+') as priors_file:
                word_alignment.make_priors([], priors_file.name)

    def test_make_priors_with_scores(self):
        """Test making priors with the score file option

        Note: The result of the alignment is not deterministic and the
        test could fail with really bad luck.

        """
        prior_data1 = ['%s.' % ('ab ' * (line + 1)) for line in range(10)] * 5
        prior_data2 = ['%s.' % ('AB ' * (line + 1)) for line in range(10)] * 5
        with tempfile.NamedTemporaryFile('w+') as priors_file, tempfile.NamedTemporaryFile('w+') as score_file:
            word_alignment.make_priors(zip(prior_data1, prior_data2), priors_file.name, score_file=score_file.name)
            self.assertTrue(priors_file.read())
            scores = [json.loads(line) for line in score_file]
            self.assertEqual(len(scores), len(prior_data1))

    def test_scoring_priors(self):
        """Test word alignment scoring on artificial data using priors

        Note: The result of the alignment is not deterministic and the
        test could fail with really bad luck.

        """
        prior_data1 = ['%s.' % ('ab ' * (line + 1)) for line in range(10)] * 5
        prior_data2 = ['%s.' % ('AB ' * (line + 1)) for line in range(10)] * 5
        data1 = ['%s.' % ('ab ' * (line + 1)) for line in range(5)] + ['', 'ab ab ab ab ab ab .']
        data2 = ['%s.' % ('AB ' * (line + 1)) for line in range(5)] + ['', 'AB']
        with tempfile.NamedTemporaryFile('w+') as priors_file:
            word_alignment.make_priors(zip(prior_data1, prior_data2), priors_file.name)
            align_filter = word_alignment.WordAlignFilter(src_threshold=0, tgt_threshold=0, priors=priors_file.name)
            scores = []
            bools = []
            for score in align_filter.score(zip(data1, data2)):
                scores.append(score)
                bools.append(align_filter.accept(score))
            logging.info(scores)
            self.assertSequenceEqual(bools, [True] * 5 + [True, False])

    def test_filtering(self):
        """Test word alignment filtering on artificial data

        Note: The result of the alignment is not deterministic and the
        test could fail with really bad luck.

        """
        data1 = ['%s.' % ('ab ' * (line + 1)) for line in range(10)] * 5 + ['ab ab ab ab ab ab .', ''] * 2
        data2 = ['%s.' % ('AB ' * (line + 1)) for line in range(10)] * 5 + ['AB', ''] * 2
        logging.info(data1)
        logging.info(data2)
        align_filter = word_alignment.WordAlignFilter(src_threshold=0, tgt_threshold=0)
        scores = []
        bools = []
        filtered = list(align_filter.filter(zip(data1, data2)))
        logging.info(filtered)
        self.assertSequenceEqual(filtered, list(zip(data1, data2))[:50] + [('', '')] * 2)

    def test_filtering_with_tokenization(self):
        """Test word alignment filtering on artificial data with tokenization

        Note: The result of the alignment is not deterministic and the
        test could fail with really bad luck.

        """
        data1 = ['%s.' % ('ab ' * (line + 1)).strip() for line in range(10)] * 5 + ['ab ab ab ab ab ab.', ''] * 2
        data2 = ['%s.' % ('AB ' * (line + 1)).strip() for line in range(10)] * 5 + ['AB.', ''] * 2
        logging.info(data1)
        logging.info(data2)
        align_filter = word_alignment.WordAlignFilter(
            src_threshold=0, tgt_threshold=0, src_tokenizer=('moses', 'en'), tgt_tokenizer=('moses', 'en'))
        scores = []
        bools = []
        filtered = list(align_filter.filter(zip(data1, data2)))
        logging.info(filtered)
        self.assertSequenceEqual(filtered, list(zip(data1, data2))[:50] + [('', '')] * 2)
