"""Word alignment filtering"""

import json
import logging
import os
import subprocess
import tempfile

from . import FilterABC, OpusFilterRuntimeError
from . import tokenization
from .util import file_open

logger = logging.getLogger(__name__)


EFLOMAL_PATH = os.environ.get('EFLOMAL_PATH')
if EFLOMAL_PATH is None:
    logger.warning("Please set enviroment variable EFLOMAL_PATH to use word alignment scores")
    EFLOMAL_PATH = '.'


def create_align_input_file(sentence_pairs, src_tokenizer=None, tgt_tokenizer=None):
    """Write sentence pairs to a named temporary file and return the file"""
    src_tokenize = tokenization.get_tokenize(src_tokenizer)
    tgt_tokenize = tokenization.get_tokenize(tgt_tokenizer)
    inputfile = tempfile.NamedTemporaryFile('w+')  # pylint: disable=R1732
    rawfile = tempfile.NamedTemporaryFile('w+') if src_tokenizer or tgt_tokenizer else None  # pylint: disable=R1732
    empty = []
    n_non_empty = 0
    for idx, pair in enumerate(sentence_pairs):
        if len(pair) != 2:
            raise ValueError("Only bilingual input supported by WordAlignFilter")
        if all(not sentence for sentence in pair):
            empty.append(idx)
            continue
        sent1, sent2 = pair
        if rawfile:
            rawfile.write(f'{sent1} ||| {sent2}\n')
        inputfile.write(f'{src_tokenize(sent1)} ||| {tgt_tokenize(sent2)}\n')
        n_non_empty += 1
    if rawfile:
        rawfile.flush()
    inputfile.flush()
    empty.reverse()
    return inputfile, rawfile, empty, n_non_empty


def _run_eflomal_align(input_file, fwd_file, rev_file, model=3, priors=None,
                       scores_fwd_file=None, scores_rev_file=None):
    """Run eflomal alignment and produce alignment files"""
    priors_arg = f'--priors {priors}' if priors else ''
    scores_arg = f'-F {scores_fwd_file} -R {scores_rev_file}' if (scores_fwd_file and scores_rev_file) else ''
    command = '{path}/align.py --overwrite -i {input} -f {fwd} -r {rev} {scores} --model {model} -M {model} {priors}'.format(
        path=EFLOMAL_PATH, input=input_file, fwd=fwd_file, rev=rev_file, scores=scores_arg,
        model=model, priors=priors_arg)
    return subprocess.run(command.split(), check=True)


def _run_eflomal_scoring(input_file, scores_fwd_file, scores_rev_file,
                         model=3, priors=None):
    """Run eflomal alignment and produce score files"""
    priors_arg = f'--priors {priors}' if priors else ''
    command = '{path}/align.py -i {input} -F {fwd} -R {rev} --model {model} -M {model} {priors}'.format(
        path=EFLOMAL_PATH, input=input_file, fwd=scores_fwd_file, rev=scores_rev_file,
        model=model, priors=priors_arg)
    return subprocess.run(command.split(), check=True)


def _run_eflomal_priors(input_file, scores_fwd_file, scores_rev_file, priors_file):
    """Run eflomal prior estimation"""
    command = f'{EFLOMAL_PATH}/makepriors.py -i {input_file} -f {scores_fwd_file} -r {scores_rev_file} --priors {priors_file}'
    return subprocess.run(command.split(), check=True)


def eflomal_to_opusfilter_scores(scores_fwd_file, scores_rev_file, output_score_file):
    """Write OpusFilter score file (JSONLines) from eflomal score files"""
    with file_open(output_score_file, 'w') as fobj:
        for score1, score2 in zip(scores_fwd_file, scores_rev_file):
            obj = {WordAlignFilter.__name__: [float(score1.strip()), float(score2.strip())]}
            fobj.write(json.dumps(obj, sort_keys=True) + '\n')


def make_priors(sentence_pairs, priors_file, model=3, score_file=None):
    """Create alignment priors from clean sentence pairs"""
    input_file, _, _, num = create_align_input_file(sentence_pairs)
    if num == 0:
        raise OpusFilterRuntimeError("No training data available for word alignment priors")
    with tempfile.NamedTemporaryFile('w+') as fwd_file, tempfile.NamedTemporaryFile('w+') as rev_file, \
         tempfile.NamedTemporaryFile('w+') as scores_fwd_file, tempfile.NamedTemporaryFile('w+') as scores_rev_file:
        process = _run_eflomal_align(
            input_file.name, fwd_file.name, rev_file.name, model=model, priors=None,
            scores_fwd_file=scores_fwd_file.name if score_file else None,
            scores_rev_file=scores_rev_file.name if score_file else None)
        process.check_returncode()
        if score_file:
            eflomal_to_opusfilter_scores(scores_fwd_file, scores_rev_file, score_file)
        process = _run_eflomal_priors(input_file.name, fwd_file.name, rev_file.name, priors_file)
        process.check_returncode()
        input_file.close()


class WordAlignFilter(FilterABC):
    """Filtering based on eflomal word aligment scores

    See :cite:`ostling-tiedemann-2016-efficient`

    """

    _empty_pair_sentinel = object()

    def __init__(self, src_threshold=0, tgt_threshold=0, priors=None, model=3,
                 src_tokenizer=None, tgt_tokenizer=None, score_for_empty=-100, **kwargs):
        super().__init__(**kwargs)
        self.src_threshold = src_threshold
        self.tgt_threshold = tgt_threshold
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.priors = os.path.join(self.workdir, priors) if priors else None
        self.model = model
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

    def score(self, pairs):
        input_file, raw_file, empty_pairs, _ = create_align_input_file(
            pairs, src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer)
        if raw_file:
            raw_file.close()
        with tempfile.NamedTemporaryFile('w+') as scores_fwd_file, \
                tempfile.NamedTemporaryFile('w+') as scores_rev_file:
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

    def accept(self, score):
        return score[0] < self.src_threshold and score[1] < self.tgt_threshold

    def _filtergen(self, pairs, filterfalse=False, decisions=False):
        """Filter or yield decisions for filtering"""
        input_file, raw_file, empty_pairs, _ = create_align_input_file(
            pairs, src_tokenizer=self.src_tokenizer, tgt_tokenizer=self.tgt_tokenizer)
        with tempfile.NamedTemporaryFile('w+') as scores_fwd_file, \
                tempfile.NamedTemporaryFile('w+') as scores_rev_file:
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
            for sent1, sent2, score in self._get_segments_and_score(
                    output_file, scores_fwd_file, scores_rev_file, empty_pairs):
                if decisions:
                    yield self.accept(score)
                elif bool(filterfalse) != bool(self.accept(score)):
                    yield sent1, sent2
            if raw_file:
                raw_file.close()
            input_file.close()

    def _get_segments_and_score(self, output_file, scores_fwd_file, scores_rev_file, empty_pairs):
        """Combine input segments and scores with separately collected empty input pairs"""
        for item in self._with_empty_pairs(
                zip(output_file, scores_fwd_file, scores_rev_file), empty_pairs):
            if item == self._empty_pair_sentinel:
                score = [self.score_for_empty, self.score_for_empty]
                sent1, sent2 = '', ''
            else:
                pair, line1, line2 = item
                score = [float(line1.strip()), float(line2.strip())]
                sent1, sent2 = pair.strip().split(' ||| ')
            yield sent1, sent2, score

    def filter(self, pairs):
        return self._filtergen(pairs, decisions=False, filterfalse=False)

    def filterfalse(self, pairs):
        return self._filtergen(pairs, decisions=False, filterfalse=True)

    def decisions(self, pairs):
        return self._filtergen(pairs, decisions=True, filterfalse=False)
