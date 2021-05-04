"""Word alignment filtering"""

import logging
import os
import subprocess
import tempfile

from . import FilterABC
from . import tokenization

logger = logging.getLogger(__name__)


EFLOMAL_PATH = os.environ.get('EFLOMAL_PATH')
if EFLOMAL_PATH is None:
    logger.warning("Please set enviroment variable EFLOMAL_PATH to use word alignment scores")
    EFLOMAL_PATH = '.'


def create_align_input_file(sentence_pairs, src_tokenizer=None, tgt_tokenizer=None):
    """Write sentence pairs to a named temporary file and return the file"""
    src_tokenize = tokenization.get_tokenize(src_tokenizer)
    tgt_tokenize = tokenization.get_tokenize(tgt_tokenizer)
    inputfile = tempfile.NamedTemporaryFile('w+')
    rawfile = tempfile.NamedTemporaryFile('w+') if src_tokenizer or tgt_tokenizer else None
    empty = []
    for idx, pair in enumerate(sentence_pairs):
        if len(pair) != 2:
            raise ValueError("Only bilingual input supported by WordAlignFilter")
        if all(not sentence for sentence in pair):
            empty.append(idx)
            continue
        sent1, sent2 = pair
        if rawfile:
            rawfile.write('{} ||| {}\n'.format(sent1, sent2))
        inputfile.write('{} ||| {}\n'.format(src_tokenize(sent1), tgt_tokenize(sent2)))
    if rawfile:
        rawfile.flush()
    inputfile.flush()
    empty.reverse()
    return inputfile, rawfile, empty


def _run_eflomal_align(input_file, fwd_file, rev_file, model=3, priors=None):
    """Run eflomal alignment and produce alignment files"""
    priors_arg = '--priors {}'.format(priors) if priors else ''
    command = '{path}/align.py --overwrite -i {input} -f {fwd} -r {rev} --model {model} -M {model} {priors}'.format(
        path=EFLOMAL_PATH, input=input_file, fwd=fwd_file, rev=rev_file,
        model=model, priors=priors_arg)
    return subprocess.run(command.split())


def _run_eflomal_scoring(input_file, scores_fwd_file, scores_rev_file,
                         model=3, priors=None):
    """Run eflomal alignment and produce score files"""
    priors_arg = '--priors {}'.format(priors) if priors else ''
    command = '{path}/align.py -i {input} -F {fwd} -R {rev} --model {model} -M {model} {priors}'.format(
        path=EFLOMAL_PATH, input=input_file, fwd=scores_fwd_file, rev=scores_rev_file,
        model=model, priors=priors_arg)
    return subprocess.run(command.split())


def _run_eflomal_priors(input_file, scores_fwd_file, scores_rev_file, priors_file):
    """Run eflomal prior estimation"""
    command = '{path}/makepriors.py -i {input} -f {fwd} -r {rev} --priors {priors}'.format(
        path=EFLOMAL_PATH, input=input_file, fwd=scores_fwd_file, rev=scores_rev_file,
        priors=priors_file)
    return subprocess.run(command.split())


def make_priors(sentence_pairs, priors_file, model=3):
    """Create alignment priors from clean sentence pairs"""
    input_file, _, _ = create_align_input_file(sentence_pairs)
    fwd_file = tempfile.NamedTemporaryFile('w+')
    rev_file = tempfile.NamedTemporaryFile('w+')
    process = _run_eflomal_align(
        input_file.name, fwd_file.name, rev_file.name, model=model, priors=None)
    process.check_returncode()
    process = _run_eflomal_priors(input_file.name, fwd_file.name, rev_file.name, priors_file)
    process.check_returncode()
    input_file.close()
    fwd_file.close()
    rev_file.close()


class WordAlignFilter(FilterABC):
    """Filtering based on eflomal word aligment scores"""

    _empty_pair_sentinel = object()

    def __init__(self, src_threshold=0, tgt_threshold=0, priors=None, model=3,
                 src_tokenizer=None, tgt_tokenizer=None, score_for_empty=-100, **kwargs):
        self.src_threshold = src_threshold
        self.tgt_threshold = tgt_threshold
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.priors = priors
        self.model = model
        self.score_for_empty = score_for_empty
        super().__init__(**kwargs)

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

    def score(self, pairs):
        input_file, raw_file, empty_pairs = create_align_input_file(
            pairs, src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer)
        if raw_file:
            raw_file.close()
        scores_fwd_file = tempfile.NamedTemporaryFile('w+')
        scores_rev_file = tempfile.NamedTemporaryFile('w+')
        process = _run_eflomal_scoring(input_file.name, scores_fwd_file.name, scores_rev_file.name,
                                       model=self.model, priors=self.priors)
        process.check_returncode()
        scores_fwd_file.seek(0)
        scores_rev_file.seek(0)
        for item in self._with_empty_pairs(
                zip(scores_fwd_file, scores_rev_file), empty_pairs):
            if item == self._empty_pair_sentinel:
                yield [self.score_for_empty, self.score_for_empty]
            else:
                line1, line2 = item
                yield [float(line1.strip()), float(line2.strip())]
        input_file.close()
        scores_fwd_file.close()
        scores_rev_file.close()

    def accept(self, score):
        return score[0] < self.src_threshold and score[1] < self.tgt_threshold

    def _filtergen(self, pairs, filterfalse=False, decisions=False):
        input_file, raw_file, empty_pairs = create_align_input_file(
            pairs, src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer)
        scores_fwd_file = tempfile.NamedTemporaryFile('w+')
        scores_rev_file = tempfile.NamedTemporaryFile('w+')
        process = _run_eflomal_scoring(input_file.name, scores_fwd_file.name, scores_rev_file.name,
                                       model=self.model, priors=self.priors)
        process.check_returncode()
        if raw_file:
            # input_file contains tokenized text, use raw_file
            output_file = raw_file
        else:
            output_file = input_file
        output_file.seek(0)
        scores_fwd_file.seek(0)
        scores_rev_file.seek(0)
        for item in self._with_empty_pairs(
                zip(output_file, scores_fwd_file, scores_rev_file), empty_pairs):
            if item == self._empty_pair_sentinel:
                score = [self.score_for_empty, self.score_for_empty]
                sent1, sent2 = '', ''
            else:
                pair, line1, line2 = item
                score = [float(line1.strip()), float(line2.strip())]
                sent1, sent2 = pair.strip().split(' ||| ')
            if decisions:
                yield self.accept(score)
            elif bool(filterfalse) != bool(self.accept(score)):
                yield sent1, sent2
        if raw_file:
            raw_file.close()
        input_file.close()
        scores_fwd_file.close()
        scores_rev_file.close()

    def filter(self, pairs):
        return self._filtergen(pairs, decisions=False, filterfalse=False)

    def filterfalse(self, pairs):
        return self._filtergen(pairs, decisions=False, filterfalse=True)

    def decisions(self, pairs):
        return self._filtergen(pairs, decisions=True, filterfalse=False)
