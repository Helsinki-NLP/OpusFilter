"""Sentence embeddings"""

import collections
import itertools
import logging
import os
import pickle

from tqdm import tqdm

from . import FilterABC, ConfigurationError
from .util import file_open, grouper


logger = logging.getLogger(__name__)


class ParallelNearestNeighbors:
    """Wrapper for sklearn.neighbors.NearestNeighbors"""

    def __init__(self, input_files, languages, embedding_model=None, n_neighbors=4,
                 algorithm='brute', metric='cosine'):
        try:
            from laserembeddings import Laser
        except ImportError:
            logger.warning("Could not load laserembeddings, LASER filtering not supported")
            raise
        if len(input_files) != len(languages):
            raise ConfigurationError(
                f"The number of input files does not match to the number of languages: {input_files} {languages}")
        self.languages = languages
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.embedding_model = embedding_model if embedding_model else Laser()
        self.embeddings, self.nn_model = self.fit_neighborhoods(input_files)

    def fit_neighborhoods(self, input_files):
        """Fit the neighborhood model"""
        from sklearn.neighbors import NearestNeighbors
        models = {}
        for language, fname in zip(self.languages, input_files):
            logger.info("Training neighbor model for %s", language)
            with file_open(fname, 'r') as ldata:
                segments = ldata.readlines()
            logger.info("Collecting embeddings...")
            embeddings = self.embedding_model.embed_sentences(tqdm(segments), lang=language)
            model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm,
                                     metric=self.metric, n_jobs=-1)
            logger.info("Fitting model...")
            model.fit(embeddings)
            models[language] = model
        return embeddings, models

    def query(self, segments, language, n_neighbors=None, return_distance=True):
        """Return nearest neighbors for the segments in language"""
        embeddings = self.embedding_model.embed_sentences(segments, lang=language)
        return self.nn_model[language].kneighbors(
            embeddings, n_neighbors=n_neighbors, return_distance=return_distance)


class SentenceEmbeddingFilter(FilterABC):
    """Filtering based on multilingual sentence embeddings

    For details of the method, see :cite:`artetxe-schwenk-2018-margin`
    and :cite:`chaudhary-etal-2019-low`.

    """

    def __init__(self, languages=None, threshold=0.5, nn_model=None, chunksize=200, **kwargs):
        try:
            from laserembeddings import Laser
        except ImportError:
            logger.warning("Could not load laserembeddings, LASER filtering not supported")
            raise
        if languages is None:
            raise ConfigurationError("A list of language codes needs to be defined")
        self.threshold = threshold
        self.languages = languages
        self.embedding_model = Laser()
        self.nn_model = None
        self.chunksize = chunksize
        super().__init__(**kwargs)
        if nn_model:
            with open(os.path.join(self.workdir, nn_model), 'rb') as fobj:
                self.nn_model = pickle.load(fobj)

    def _cosine_similarities(self, pairs):
        """Calculate cosine similarities for the segments"""
        from scipy.spatial.distance import cosine
        chunksize = len(pairs)
        segments = [segment for pair in pairs for segment in pair]
        languages = self.languages * chunksize
        embeddings = self.embedding_model.embed_sentences(segments, languages)
        for pos in range(0, len(languages), len(self.languages)):
            yield [1 - cosine(embeddings[pos + idx1, :], embeddings[pos + idx2, :])
                   for idx1, idx2 in itertools.combinations(range(len(self.languages)), 2)]

    @staticmethod
    def _ratio_normalize(vec1, vec2, n_neighbors, nn_sum1, nn_sum2):
        """Cosine similarity normalized by similarity to nearest neighbors"""
        from scipy.spatial.distance import cosine
        return 2 * n_neighbors * (1 - cosine(vec1, vec2)) / (nn_sum1 + nn_sum2)

    def _normalized_similarities(self, pairs):
        """Calculate normalized cosine similarities for the segments"""
        chunksize = len(pairs)
        n_neighbors = self.nn_model.n_neighbors
        input_per_lang = zip(*pairs)
        output_per_lang = []
        nn_sums = collections.defaultdict(dict)
        for idx, segments in enumerate(input_per_lang):
            for other_idx, other_language in enumerate(self.languages):
                if idx == other_idx:
                    continue
                dists, _ = self.nn_model.query(segments, other_language)
                nn_sums[idx][other_idx] = dists.sum(axis=1)
            embeddings = self.embedding_model.embed_sentences(segments, self.languages[idx])
            output_per_lang.append(embeddings)
        for pos in range(chunksize):
            yield [self._ratio_normalize(output_per_lang[idx1][pos, :], output_per_lang[idx2][pos, :],
                                         n_neighbors, nn_sums[idx1][idx2][pos], nn_sums[idx2][idx1][pos])
                   for idx1, idx2 in itertools.combinations(range(len(self.languages)), 2)]

    def _score_chunk(self, chunk):
        """Return scores for a chunk of data"""
        return self._cosine_similarities(chunk) if self.nn_model is None else \
            self._normalized_similarities(chunk)

    def score(self, pairs):
        for chunk in grouper(pairs, self.chunksize):
            return self._score_chunk(chunk)

    def accept(self, score):
        return all(similarity >= self.threshold for similarity in score)

    def filter(self, pairs):
        for chunk in grouper(pairs, self.chunksize):
            for pair, score in zip(pairs, self._score_chunk(chunk)):
                if self.accept(score):
                    yield pair
