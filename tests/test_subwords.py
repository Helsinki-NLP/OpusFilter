import logging
import tempfile
import unittest

from opusfilter import subwords, ConfigurationError


class TestBPESegmentation(unittest.TestCase):

    data = ['koira jahtasi kissaa',
            'kissa kiipesi puuhun',
            'puu huojui tuulessa',
            'koira haukkui maassa ja kissa sähisi puussa',
            'melu peittyi tuuleen']

    def _get_segmenter(self):
        with tempfile.NamedTemporaryFile('w+') as datafile, tempfile.NamedTemporaryFile('w+') as modelfile:
            for line in self.data:
                datafile.write(line + '\n')
            datafile.seek(0)
            subwords.BPESegmentation.train(datafile.name, modelfile.name, 20, min_frequency=1, num_workers=1)
            return subwords.BPESegmentation(modelfile.name)

    def test_segmentation(self):
        segmenter = self._get_segmenter()
        self.assertEqual(segmenter.get_subwords('koira'), ['koira'])
        self.assertEqual(segmenter.get_subwords('vai'), ['v', 'a', 'i'])
        output = []
        for segment in self.data:
            out = segmenter.split(segment)
            self.assertGreater(len(out.split()), len(segment.split()))
            output.append(out)
        joined = [segmenter.join(segment) for segment in output]
        self.assertSequenceEqual(self.data, joined)


class TestMorfessorSegmentation(TestBPESegmentation):

    def _get_segmenter(self):
        with tempfile.NamedTemporaryFile('w+') as datafile, tempfile.NamedTemporaryFile('w+') as modelfile:
            for line in self.data:
                datafile.write(line + '\n')
            datafile.seek(0)
            subwords.MorfessorSegmentation.train(
                datafile.name, modelfile.name, corpusweight=1.0, min_frequency=1, dampening=None, seed=0)
            return subwords.MorfessorSegmentation(modelfile.name)
