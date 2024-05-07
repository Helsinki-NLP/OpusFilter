import logging
import os
import pickle
import requests
import shutil
import tempfile
import unittest
from unittest import mock

from opusfilter import ConfigurationError
from opusfilter.embeddings import *
from opusfilter.pipeline import FilterPipeline


try:
    import laserembeddings
except ImportError:
    logging.warning("Could not load laserembeddings, LASER filtering not supported")


def mocked_score_chunk(obj, chunk):
    """Return scores for a chunk of data"""
    mocked_score_chunk.counter += 1
    return obj._cosine_similarities(chunk) if obj.nn_model is None else \
        obj._normalized_similarities(chunk)

mocked_score_chunk.counter = 0


@unittest.skipIf('laserembeddings' not in globals(), 'laserembeddings package not installed')
class TestSentenceEmbeddingFilter(unittest.TestCase):

    bi_inputs = [
        ('Hei maailma!', 'Hello world!'),
        ('Niityllä kasvaa kukkia.', 'Flowers grow in the meadow.'),
        ('Niityllä kasvaa kukkia.', 'Hello world!'),
        ('Hei maailma!', 'Flowers grow in the meadow.')
    ]
    bi_langs = ['fi', 'en']

    ref_inputs = [
        ('Hei!', 'Hi!'),
        ('Terve kaikki.', 'Greetings everyone.'),
        ('Maailman kaunein kukka', 'The most beautiful flower in the world'),
        ('Viisitoista kukkaruukkua', 'Fifteen plant pots'),
        ('Iso musta kissa istui ikkunalaudalla.', 'A big black cat sat on a sill.'),
        ('Koska päästään syömään?', 'When will we get to eat?'),
        ('Kuva vastaa tuhatta sanaa.', 'A picture is worth thousand words.'),
        ('Terve, ja kiitos kaloista', 'So long, and thanks for all the fish'),
    ]

    def _train_nn_model(self):
        with tempfile.NamedTemporaryFile('w+') as ref_fi, \
             tempfile.NamedTemporaryFile('w+') as ref_en:
            for seg_fi, seg_en in self.ref_inputs:
                ref_fi.write(seg_fi + '\n')
                ref_en.write(seg_en + '\n')
            ref_fi.seek(0)
            ref_en.seek(0)
            nn_model = ParallelNearestNeighbors([ref_fi.name, ref_en.name], ['fi', 'en'])
        return nn_model

    def test_train_nn_model(self):
        nn_model = self._train_nn_model()
        dist, ind = nn_model.query(self.bi_inputs[0][1], 'fi')
        self.assertEqual(ind[0][0], 0)
        dist, ind = nn_model.query([pair[1] for pair in self.bi_inputs], 'en', n_neighbors=2)
        self.assertEqual(ind.shape, (4, 2))

    @mock.patch('opusfilter.embeddings.SentenceEmbeddingFilter._score_chunk', mocked_score_chunk)
    def test_bilingual_score(self):
        mocked_score_chunk.counter = 0
        testfilter = SentenceEmbeddingFilter(languages=self.bi_langs, threshold=0.4, chunksize=2)
        expected = [True, True, False, False]
        results = [testfilter.accept(x) for x in testfilter.score(self.bi_inputs)]
        for result, correct in zip(results, expected):
            self.assertEqual(result, correct)
        self.assertEqual(mocked_score_chunk.counter, (len(self.bi_inputs) + 1) // 2)

    @mock.patch('opusfilter.embeddings.SentenceEmbeddingFilter._score_chunk', mocked_score_chunk)
    def test_bilingual_filter(self):
        mocked_score_chunk.counter = 0
        testfilter = SentenceEmbeddingFilter(languages=self.bi_langs, threshold=0.4, chunksize=2)
        expected = [self.bi_inputs[0], self.bi_inputs[1]]
        results = testfilter.filter(self.bi_inputs)
        for result, correct in zip(results, expected):
            self.assertEqual(result, correct)
        self.assertEqual(mocked_score_chunk.counter, (len(self.bi_inputs) + 1) // 2)

    @mock.patch('opusfilter.embeddings.SentenceEmbeddingFilter._score_chunk', mocked_score_chunk)
    def test_bilingual_filterfalse(self):
        mocked_score_chunk.counter = 0
        testfilter = SentenceEmbeddingFilter(languages=self.bi_langs, threshold=0.4, chunksize=2)
        expected = [self.bi_inputs[2], self.bi_inputs[3]]
        logging.warning(expected)
        results = list(testfilter.filterfalse(self.bi_inputs))
        logging.warning(results)
        for result, correct in zip(results, expected):
            self.assertEqual(result, correct)
        self.assertEqual(mocked_score_chunk.counter, (len(self.bi_inputs) + 1) // 2)

    def test_bilingual_margin_ratios(self):
        nn_model = self._train_nn_model()
        with tempfile.NamedTemporaryFile('w+b') as model_file:
            pickle.dump(nn_model, model_file)
            model_file.seek(0)
            testfilter = SentenceEmbeddingFilter(languages=self.bi_langs, threshold=1.0, nn_model=model_file.name)
        expected = [True, True, False, False]
        results = [testfilter.accept(x) for x in testfilter.score(self.bi_inputs)]
        for result, correct in zip(results, expected):
            self.assertEqual(result, correct)

    def test_pipeline_chunking(self):
        testfilter = SentenceEmbeddingFilter(languages=self.bi_langs, threshold=0.4, chunksize=19)
        inputs = 50 * self.bi_inputs
        expected = 50 * [True, True, False, False]
        results = [testfilter.accept(x) for x in testfilter.score(inputs)]
        for result, correct in zip(results, expected):
            self.assertEqual(result, correct)
        pipeline = FilterPipeline(filters=[testfilter])
        pipeline.chunksize = 30
        filtered = list(pipeline.filter(inputs))
        self.assertEqual(len(filtered), len([x for x in expected if x]))
        scores = list(pipeline.score(inputs))
        self.assertEqual(len(scores), len(expected))
