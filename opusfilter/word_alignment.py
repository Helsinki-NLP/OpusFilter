"""Word alignment filtering"""

import contextlib
import json
import logging
import math
import os
import tempfile

from . import FilterABC, OpusFilterRuntimeError, CLEAN_LOW
from . import tokenization
from .util import file_open

logger = logging.getLogger(__name__)


def eflomal_to_opusfilter_scores(scores_fwd_file, scores_rev_file, output_score_file):
    """Write OpusFilter score file (JSONLines) from eflomal score files"""
    with file_open(output_score_file, 'w') as fobj:
        for score1, score2 in zip(scores_fwd_file, scores_rev_file):
            obj = {WordAlignFilter.__name__: [float(score1.strip()), float(score2.strip())]}
            fobj.write(json.dumps(obj, sort_keys=True) + '\n')


def sentence_generator(filename, tokenizer=None):
    """Yield and optionally tokenize sentences from given file"""
    tokenize = tokenization.get_tokenize(tokenizer)
    num = 0
    with file_open(filename) as fobj:
        for line in fobj:
            yield tokenize(line.rstrip())
            num += 1
    if num == 0:
        raise OpusFilterRuntimeError("No training data available for word alignment priors")


def make_priors(src_file, tgt_file, priors_file, model=3, score_file=None, src_tokenizer=None, tgt_tokenizer=None):
    """Create alignment priors from clean sentence pairs"""
    try:
        import eflomal
    except ImportError:
        logger.warning("Could not load eflomal, word alignment filtering not supported")
        raise
    aligner = eflomal.Aligner(model=model)
    with tempfile.NamedTemporaryFile('w+') as fwd_file, tempfile.NamedTemporaryFile('w+') as rev_file, \
         tempfile.NamedTemporaryFile('w+') as scores_fwd_file, tempfile.NamedTemporaryFile('w+') as scores_rev_file:
        aligner.align(sentence_generator(src_file, src_tokenizer), sentence_generator(tgt_file, tgt_tokenizer),
                      links_filename_fwd=fwd_file.name, links_filename_rev=rev_file.name,
                      scores_filename_fwd=scores_fwd_file.name if score_file else None,
                      scores_filename_rev=scores_rev_file.name if score_file else None, quiet=True)
        if score_file:
            eflomal_to_opusfilter_scores(scores_fwd_file, scores_rev_file, score_file)
        fwd_file.seek(0)
        rev_file.seek(0)
        priors_list = eflomal.calculate_priors(
            sentence_generator(src_file, src_tokenizer), sentence_generator(tgt_file, tgt_tokenizer),
            fwd_file, rev_file)
        with open(priors_file, 'w', encoding='utf-8') as priorsf:
            eflomal.write_priors(priorsf, *priors_list)


class WordAlignFilter(FilterABC):
    """Filtering based on eflomal word aligment scores

    See :cite:`ostling-tiedemann-2016-efficient`

    """

    score_direction = CLEAN_LOW
    accept_threshold = math.inf
    reject_threshold = -math.inf
    _empty_pair_sentinel = object()

    def __init__(self, src_threshold=0, tgt_threshold=0, priors=None, model=3,
                 src_tokenizer=None, tgt_tokenizer=None, score_for_empty=-100, **kwargs):
        try:
            import eflomal
        except ImportError:
            logger.warning("Could not load eflomal, word alignment filtering not supported")
            raise
        super().__init__(**kwargs)
        self.src_threshold = src_threshold
        self.tgt_threshold = tgt_threshold
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.priors = os.path.join(self.workdir, priors) if priors else None
        self.aligner = eflomal.Aligner(model=model)
        self.score_for_empty = score_for_empty

    def _with_empty_pairs(self, iterator, empty_pairs):
        """Append empty_pair_sentinel to the iterator positions indicated by empty_pairs"""
        idx = 0
        while True:
            if empty_pairs and empty_pairs[-1] == idx:
                # Next is empty pair
                yield self._empty_pair_sentinel
                empty_pairs.pop()
                idx += 1
                continue
            # Next is pair from iterator
            try:
                pair = next(iterator)
            except StopIteration:
                # Yield any remaining empty lines
                for _ in empty_pairs:
                    yield self._empty_pair_sentinel
                break
            yield pair
            idx += 1

    def _write_pairs(self, pairs, outfiles, raw_outfiles=None):
        tokenizers = [tokenization.get_tokenize(self.src_tokenizer),
                      tokenization.get_tokenize(self.tgt_tokenizer)]
        empty = []
        if raw_outfiles is None:
            raw_outfiles = (None, None)
        for idx, pair in enumerate(pairs):
            if len(pair) != 2:
                raise ValueError("Only bilingual input supported by WordAlignFilter")
            if all(not sentence for sentence in pair):
                empty.append(idx)
                continue
            for sent, tokenizer, outfile, raw_outfile in zip(pair, tokenizers, outfiles, raw_outfiles):
                if raw_outfile:
                    raw_outfile.write(f'{sent}\n')
                if tokenizer:
                    sent = tokenizer(sent)
                outfile.write(f'{sent}\n')
        empty.reverse()
        return empty

    def score(self, pairs):
        with contextlib.ExitStack() as stack:
            # Write input data to temporary files
            input_files = (stack.enter_context(tempfile.TemporaryFile('w+')),
                           stack.enter_context(tempfile.TemporaryFile('w+')))
            empty_pairs = self._write_pairs(pairs, input_files)
            for fobj in input_files:
                fobj.seek(0)
            # Run aligner
            score_files = (stack.enter_context(tempfile.NamedTemporaryFile('w+')),
                           stack.enter_context(tempfile.NamedTemporaryFile('w+')))
            priorsf = stack.enter_context(open(self.priors, 'r', encoding='utf-8')) if self.priors else None
            self.aligner.align(*input_files, scores_filename_fwd=score_files[0].name,
                               scores_filename_rev=score_files[1].name, priors_input=priorsf, quiet=True)
            for fobj in score_files:
                fobj.seek(0)
            # Yield results (scores)
            for item in self._with_empty_pairs(zip(*score_files), empty_pairs):
                if item == self._empty_pair_sentinel:
                    yield [self.score_for_empty, self.score_for_empty]
                else:
                    yield [float(item[0].strip()), float(item[1].strip())]

    def accept(self, score):
        return score[0] < self.src_threshold and score[1] < self.tgt_threshold

    def _filtergen(self, pairs, filterfalse=False, decisions=False):
        """Filter or yield decisions for filtering"""
        with contextlib.ExitStack() as stack:
            # Write input data to temporary files
            input_files = (stack.enter_context(tempfile.NamedTemporaryFile('w+')),
                           stack.enter_context(tempfile.NamedTemporaryFile('w+')))
            if self.src_tokenizer or self.tgt_tokenizer:
                raw_input_files = (stack.enter_context(tempfile.NamedTemporaryFile('w+')),
                                   stack.enter_context(tempfile.NamedTemporaryFile('w+')))
                empty_pairs = self._write_pairs(pairs, input_files, raw_input_files)
                output_files = raw_input_files
            else:
                empty_pairs = self._write_pairs(pairs, input_files)
                output_files = input_files
            for fobj in input_files:
                fobj.seek(0)
            # Run aligner
            score_files = (stack.enter_context(tempfile.NamedTemporaryFile('w+')),
                           stack.enter_context(tempfile.NamedTemporaryFile('w+')))
            self.aligner.align(
                *input_files, scores_filename_fwd=score_files[0].name, scores_filename_rev=score_files[1].name,
                priors_input=stack.enter_context(open(self.priors, 'r', encoding='utf-8')) if self.priors else None,
                quiet=True)
            for fobj in (*score_files, *output_files):
                fobj.seek(0)
            # Yield results (decisions or filtered original sentences)
            for pair, score in self._get_segments_and_score(output_files, score_files, empty_pairs):
                if decisions:
                    yield self.accept(score)
                elif bool(filterfalse) != bool(self.accept(score)):
                    yield pair

    def _get_segments_and_score(self, output_files, score_files, empty_pairs):
        """Combine input segments and scores with separately collected empty input pairs"""
        for item in self._with_empty_pairs(zip(*output_files, *score_files), empty_pairs):
            if item == self._empty_pair_sentinel:
                score = [self.score_for_empty, self.score_for_empty]
                sent1, sent2 = '', ''
            else:
                sent1, sent2, line1, line2 = item
                score = [float(line1.strip()), float(line2.strip())]
            yield (sent1.strip(), sent2.strip()), score

    def filter(self, pairs):
        return self._filtergen(pairs, decisions=False, filterfalse=False)

    def filterfalse(self, pairs):
        return self._filtergen(pairs, decisions=False, filterfalse=True)

    def decisions(self, pairs):
        return self._filtergen(pairs, decisions=True, filterfalse=False)
