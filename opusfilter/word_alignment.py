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
    for sent1, sent2 in sentence_pairs:
        inputfile.write('{} ||| {}\n'.format(src_tokenize(sent1), tgt_tokenize(sent2)))
    inputfile.flush()
    return inputfile


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
    input_file = create_align_input_file(sentence_pairs)
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

    def __init__(self, src_threshold=0, tgt_threshold=0, priors=None, model=3,
                 src_tokenizer=None, tgt_tokenizer=None, **kwargs):
        self.src_threshold = src_threshold
        self.tgt_threshold = tgt_threshold
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.priors = priors
        self.model = model
        super().__init__(**kwargs)

    def score(self, pairs):
        input_file = create_align_input_file(
            pairs, src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer)
        scores_fwd_file = tempfile.NamedTemporaryFile('w+')
        scores_rev_file = tempfile.NamedTemporaryFile('w+')
        process = _run_eflomal_scoring(input_file.name, scores_fwd_file.name, scores_rev_file.name,
                                       model=self.model, priors=self.priors)
        process.check_returncode()
        scores_fwd_file.seek(0)
        scores_rev_file.seek(0)
        for line1, line2 in zip(scores_fwd_file, scores_rev_file):
            yield {'src': float(line1.strip()), 'tgt': float(line2.strip())}
        input_file.close()
        scores_fwd_file.close()
        scores_rev_file.close()

    def accept(self, score):
        src, tgt = score['src'], score['tgt']
        return src < self.src_threshold and tgt < self.tgt_threshold

    def _filtergen(self, pairs, filterfalse=False, decisions=False):
        input_file = create_align_input_file(
            pairs, src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer)
        scores_fwd_file = tempfile.NamedTemporaryFile('w+')
        scores_rev_file = tempfile.NamedTemporaryFile('w+')
        process = _run_eflomal_scoring(input_file.name, scores_fwd_file.name, scores_rev_file.name,
                                       model=self.model, priors=self.priors)
        process.check_returncode()
        input_file.seek(0)
        scores_fwd_file.seek(0)
        scores_rev_file.seek(0)
        for input_pair, line1, line2 in zip(input_file, scores_fwd_file, scores_rev_file):
            score = {'src': float(line1.strip()), 'tgt': float(line2.strip())}
            if decisions:
                yield self.accept(score)
            elif bool(filterfalse) != bool(self.accept(score)):
                sent1, sent2 = input_pair.strip().split(' ||| ')
                yield sent1, sent2
        input_file.close()
        scores_fwd_file.close()
        scores_rev_file.close()

    def filter(self, pairs):
        return self._filtergen(pairs, decisions=False, filterfalse=False)

    def filterfalse(self, pairs):
        return self._filtergen(pairs, decisions=False, filterfalse=True)

    def decisions(self, pairs):
        return self._filtergen(pairs, decisions=True, filterfalse=False)
