
import argparse
import logging
import os
import tempfile
import unittest

from opusfilter import word_alignment


class TestAlignFilter(unittest.TestCase):

    def test_simple(self):
        """Test alignment on artificial data

        Note: The result of the alignment is not deterministic and the
        test could fail with really bad luck.

        """
        data1 = ['%s.' % ('ab ' * (line + 1)) for line in range(10)] * 5 + ['ab ab ab .']
        data2 = ['%s.' % ('AB ' * (line + 1)) for line in range(10)] * 5 + ['AB']
        logging.info(data1)
        logging.info(data2)
        align_filter = word_alignment.WordAlignFilter(src_threshold=0, tgt_threshold=0)
        scores = []
        bools = []
        for score in align_filter.score(zip(data1, data2)):
            scores.append(score)
            bools.append(align_filter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True] * 50 + [False])

    def test_priors(self):
        """Test alignment on artificial data using priors

        Note: The result of the alignment is not deterministic and the
        test could fail with really bad luck.

        """
        prior_data1 = ['%s.' % ('ab ' * (line + 1)) for line in range(10)] * 5
        prior_data2 = ['%s.' % ('AB ' * (line + 1)) for line in range(10)] * 5
        data1 = ['%s.' % ('ab ' * (line + 1)) for line in range(5)] + ['ab ab ab .']
        data2 = ['%s.' % ('AB ' * (line + 1)) for line in range(5)] + ['AB']
        priors_file = tempfile.NamedTemporaryFile('w+')
        word_alignment.make_priors(zip(prior_data1, prior_data2), priors_file.name)
        align_filter = word_alignment.WordAlignFilter(src_threshold=0, tgt_threshold=0, priors=priors_file.name)
        scores = []
        bools = []
        for score in align_filter.score(zip(data1, data2)):
            scores.append(score)
            bools.append(align_filter.accept(score))
        logging.info(scores)
        self.assertSequenceEqual(bools, [True] * 5 + [False])
        priors_file.close()
