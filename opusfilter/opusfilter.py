"""Processor for filter configurations"""

import copy
import logging
import os
import random

import json
import numpy as np
from tqdm import tqdm

from opustools import OpusRead
from opustools.util import file_open

from . import ConfigurationError
from . import pipeline
from . import lm
from . import word_alignment
from . import tokenization
from . import classifier


logger = logging.getLogger(__name__)


def dict_get(key, dictionary):
    """Recursive get for multi-part key with dot (.) as separator

    Example:
    dict_get("foo.bar", {"foo": {"bar": 5}}) -> 5

    Raises KeyError for missing keys.

    """
    parts = key.split('.')
    first = parts.pop(0)
    value = dictionary[first]
    return value if not len(parts) else dict_get('.'.join(parts), value)


class OpusFilter:
    """Apply filters to language data"""

    def __init__(self, configuration):
        self.configuration = configuration
        self.output_dir = configuration.get('common', {}).get('output_directory')
        if not self.output_dir:
            logger.warning(
                'Output directory not specified. Writing files to current '
                'directory.')
            self.output_dir = '.'
        elif not os.path.isdir(self.output_dir):
            logger.warning(
                'Directory "{}" does not exist. It will be '
                'created.'.format(self.output_dir))
            os.mkdir(self.output_dir)

        self.step_functions = {
            'opus_read': self.read_from_opus,
            'filter': self.filter_data,
            'concatenate': self.concatenate,
            'subset': self.get_subset,
            'train_ngram': self.train_ngram,
            'train_alignment': self.train_alignment,
            'score': self.score_data,
            'classify': self.classify,
            'join': self.join_scores,
            'sort': self.sort_files
        }

    def execute_steps(self, overwrite=False, last=None):
        """Execute steps in the same order as they are in the configuration"""
        for num, step in enumerate(self.configuration['steps']):
            if last is not None and num + 1 > last:
                logger.info('Stopping after step %s', last)
                break
            logger.info('Running step %s: %s', num + 1, step)
            self.step_functions[step['type']](step['parameters'], overwrite=overwrite)

    def execute_step(self, num, overwrite=False):
        """Execute single step in the configuration (first = 1, last = -1)

        Does not check any dependencies and may fail if the input
        files do not exist.

        """
        step = self.configuration['steps'][num if num < 0 else num - 1]
        logger.info('Running step %s: %s', num, step)
        self.step_functions[step['type']](step['parameters'], overwrite=overwrite)

    def read_from_opus(self, parameters, overwrite=False):
        """Download and read a corpus from OPUS"""
        src_out = os.path.join(self.output_dir, parameters['src_output'])
        tgt_out = os.path.join(self.output_dir, parameters['tgt_output'])
        if not overwrite and os.path.isfile(src_out) and os.path.isfile(tgt_out):
            logger.info("Output files exists, skipping step")
            return

        opus_reader = OpusRead(
            directory=parameters['corpus_name'],
            source=parameters['source_language'],
            target=parameters['target_language'],
            release=parameters['release'],
            preprocess=parameters['preprocessing'], write_mode='moses',
            write=[src_out, tgt_out],
            leave_non_alignments_out=True,
            download_dir=self.output_dir)

        opus_reader.printPairs()

    @staticmethod
    def pair_generator(source_file_name, target_file_name,
                       src_tokenizer=None, tgt_tokenizer=None):
        """Yield and optionally tokenize sentence pairs from given files"""
        src_tokenize = tokenization.get_tokenize(src_tokenizer)
        tgt_tokenize = tokenization.get_tokenize(tgt_tokenizer)
        with file_open(source_file_name) as source_file, \
                file_open(target_file_name) as target_file:
            for src_line in source_file:
                tgt_line = target_file.readline()
                yield (src_tokenize(src_line.rstrip()), tgt_tokenize(tgt_line.rstrip()))

    def get_pairs(self, src_filename, tgt_filename):
        """Return a generator for given sentence files"""
        source_file_name = '{result_dir}/{src_filename}'.format(
            result_dir=self.output_dir, src_filename=src_filename)
        target_file_name = '{result_dir}/{tgt_filename}'.format(
            result_dir=self.output_dir, tgt_filename=tgt_filename)
        return self.pair_generator(source_file_name, target_file_name)

    def filter_data(self, parameters, overwrite=False):
        """Write sentences to file if they pass given filters"""
        src_out = os.path.join(self.output_dir, parameters['src_output'])
        tgt_out = os.path.join(self.output_dir, parameters['tgt_output'])
        if not overwrite and os.path.isfile(src_out) and os.path.isfile(tgt_out):
            logger.info("Output files exists, skipping step")
            return
        filter_pipe = pipeline.FilterPipeline.from_config(parameters['filters'])
        filterfalse = parameters.get('filterfalse', False)
        pairs_gen = self.get_pairs(parameters['src_input'], parameters['tgt_input'])
        if filterfalse:
            pairs = filter_pipe.filterfalse(pairs_gen)
        else:
            pairs = filter_pipe.filter(pairs_gen)
        limit = parameters.get('limit')
        with file_open(src_out, 'w') as source_file, \
                file_open(tgt_out, 'w') as target_file:
            for idx, pair in tqdm(enumerate(pairs)):
                source_file.write(pair[0]+'\n')
                target_file.write(pair[1]+'\n')
                source_file.flush()
                target_file.flush()
                if limit and idx >= limit - 1:
                    break

    def concatenate(self, parameters, overwrite=False):
        """Concatenate files"""
        outfile = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(outfile):
            logger.info("Output file exists, skipping step")
            return
        with file_open(outfile, 'w') as outf:
            for infile in parameters['inputs']:
                with file_open(os.path.join(self.output_dir, infile)) as inf:
                    for line in inf:
                        outf.write(line)

    @staticmethod
    def _get_total_lines(fname):
        """Return number of lines in file"""
        with file_open(fname) as fobj:
            total = -1
            for total, _ in tqdm(enumerate(fobj)):
                pass
        return total + 1

    @staticmethod
    def _yield_subset(iterable, indices):
        """Yield items for which the indices match"""
        if not indices:
            return
        remaining = sorted(indices, reverse=True)
        cur = remaining.pop()
        for idx, item in tqdm(enumerate(iterable)):
            if idx == cur:
                yield item
                if remaining:
                    cur = remaining.pop()
                else:
                    return

    def get_subset(self, parameters, overwrite=False):
        """Get random subset of parallel data

        Keeps the order of lines, unless if shuffle_target is True in
        parameters, in which case the target lines will be in a random
        order.

        """
        src_in = os.path.join(self.output_dir, parameters['src_input'])
        tgt_in = os.path.join(self.output_dir, parameters['tgt_input'])
        src_out = os.path.join(self.output_dir, parameters['src_output'])
        tgt_out = os.path.join(self.output_dir, parameters['tgt_output'])
        if not overwrite and os.path.isfile(src_out) and os.path.isfile(tgt_out):
            logger.info("Output files exists, skipping step")
            return
        random.seed(parameters.get('seed', None))
        size = parameters['size']
        shuffle_target = parameters.get('shuffle_target', False)
        total = self._get_total_lines(src_in)
        logger.info("Sampling subset of %s lines from total %s lines", size, total)
        if shuffle_target:
            sample = random.sample(range(total), size)
            with file_open(src_in) as inf, \
                 file_open(src_out, 'w') as outf:
                for line in self._yield_subset(inf, sample):
                    outf.write(line)
            sample = random.sample(range(total), size)
            with file_open(tgt_in) as inf:
                lines = [line for line in self._yield_subset(inf, sample)]
            random.shuffle(lines)
            with file_open(tgt_out, 'w') as outf:
                for line in lines:
                    outf.write(line)
        else:
            sample = random.sample(range(total), size)
            with file_open(src_in) as inf, \
                 file_open(src_out, 'w') as outf:
                for line in self._yield_subset(inf, sample):
                    outf.write(line)
            with file_open(tgt_in) as inf, \
                 file_open(tgt_out, 'w') as outf:
                for line in self._yield_subset(inf, sample):
                    outf.write(line)

    def train_ngram(self, parameters, overwrite=False):
        """Train an n-gram language model"""
        model_out = os.path.join(self.output_dir, parameters['model'])
        if not overwrite and os.path.isfile(model_out):
            logger.info("Output file exists, skipping step")
            return
        data_name = parameters['data']
        seg_name = data_name + '.seg.gz'
        tokenizer = lm.LMTokenizer(**parameters['parameters'])
        with file_open(os.path.join(self.output_dir, data_name), 'r') as \
                infile, \
                file_open(os.path.join(self.output_dir, seg_name), 'w') as \
                outfile:
            for line in tqdm(infile):
                tokens = tokenizer.tokenize(line.strip())
                outfile.write(' '.join(tokens) + '\n')
        lm.train(os.path.join(self.output_dir, seg_name), model_out,
                 **parameters['parameters'])

    def train_alignment(self, parameters, overwrite=False):
        """Train eflomal alignment priors"""
        model_out = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(model_out):
            logger.info("Output file exists, skipping step")
            return
        pair_gen = tqdm(self.pair_generator(
            os.path.join(self.output_dir, parameters['src_data']),
            os.path.join(self.output_dir, parameters['tgt_data']),
            src_tokenizer=parameters['parameters'].get('src_tokenizer', None),
            tgt_tokenizer=parameters['parameters'].get('tgt_tokenizer', None)))
        word_alignment.make_priors(
            pair_gen, model_out, model=parameters['parameters'].get('model', 3))

    @staticmethod
    def _write_jsonl(objects, fname):
        """Write objects to file as JSON lines"""
        with file_open(fname, 'w') as fobj:
            for obj in objects:
                fobj.write(json.dumps(obj, sort_keys=True)+'\n')

    @staticmethod
    def _read_jsonl(fobj):
        """Return a generator for items in JSON lines file"""
        for line in fobj:
            yield json.loads(line)

    def score_data(self, parameters, overwrite=False):
        """Score language data based on given filters"""
        score_out = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(score_out):
            logger.info("Output file exists, skipping step")
            return
        # Make a copy so that the original paths are not modified
        filter_params = copy.deepcopy(parameters['filters'])
        for f in filter_params:
            filter_name = next(iter(f.items()))[0]
            if filter_name == 'WordAlignFilter' and 'priors' in f[filter_name]:
                f[filter_name]['priors'] = os.path.join(
                    self.output_dir, f[filter_name]['priors'])
            if filter_name == 'CrossEntropyFilter':
                src_lm_params = f[filter_name]['src_lm_params']
                src_lm_params['filename'] = os.path.join(
                    self.output_dir, src_lm_params['filename'])
                if src_lm_params.get('interpolate'):
                    for idx in range(len(src_lm_params['interpolate'])):
                        src_lm_params['interpolate'][idx][0] = os.path.join(
                            self.output_dir, src_lm_params['interpolate'][idx][0])
                tgt_lm_params = f[filter_name]['tgt_lm_params']
                tgt_lm_params['filename'] = os.path.join(
                    self.output_dir, tgt_lm_params['filename'])
                if tgt_lm_params.get('interpolate'):
                    for idx in range(len(tgt_lm_params['interpolate'])):
                        tgt_lm_params['interpolate'][idx][0] = os.path.join(
                            self.output_dir, tgt_lm_params['interpolate'][idx][0])

        pairs_gen = self.get_pairs(parameters['src_input'], parameters['tgt_input'])
        filter_pipe = pipeline.FilterPipeline.from_config(filter_params)
        scores_gen = filter_pipe.score(pairs_gen)
        self._write_jsonl(scores_gen, score_out)

    def classify(self, parameters, overwrite=False):
        """Assign cleanness probabilities to scored sentence pairs"""
        cls = classifier.FilterClassifier(**parameters)
        model, value, discard_threshold = cls.find_best_model(
                parameters['criterion'])
        cls.assign_probabilities(model)

    @staticmethod
    def _read_values(fobj, key=None, conv=None):
        """Return a generator for values in score file

        The file should either contain one JSON object per line (if
        key is not None), or single value per line. If conv is not
        None, conv(value) is yielded instead of plain value.

        """
        for line in fobj:
            val = line.rstrip() if key is None else dict_get(key, json.loads(line))
            yield val if conv is None else conv(val)

    def sort_files(self, parameters, overwrite=False):
        """Sort file(s) by values read from other file"""
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in sort")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        valuefile = parameters['values']
        reverse = parameters.get('reverse', False)
        key = parameters.get('key')
        typeconv = parameters.get('type', 'float' if key is None else None)
        if typeconv is not None:
            typeconv = {'float': float, 'int': int, 'str': str}[typeconv]
        with file_open(valuefile, 'r') as fobj:
            logger.info("Reading values from %s", valuefile)
            values = [x for x in tqdm(self._read_values(fobj, key=key, conv=typeconv))]
            order = list(np.argsort(values))
            if reverse:
                order.reverse()
        for infile, outfile in zip(infiles, outfiles):
            logger.info("Sorting file %s", infile)
            with file_open(infile, 'r') as fobj:
                lines = [line.rstrip() for line in tqdm(fobj)]
            with file_open(outfile, 'w') as fobj:
                for idx in tqdm(order):
                    fobj.write(lines[idx] + '\n')

    def join_scores(self, parameters, overwrite=False):
        """Join score files

        If a list of keys is provided, the input objects are inserted
        under the corresponding key. If the keys are not provided, or
        corresponding key is None, output object will be updated with
        the input object and existing keys will be overwritten.

        """
        def _gen(inputs, keys):
            """Generator for output objects"""
            for objects in zip(*inputs):
                new = {}
                for idx, obj in enumerate(objects):
                    if keys and keys[idx] is not None:
                        new[keys[idx]] = obj
                    else:
                        new.update(obj)
                yield new

        outfile = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(outfile):
            logger.info("Output file exists, skipping step")
            return
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        keys = parameters.get('keys')
        if keys and len(keys) != len(infiles):
            raise ConfigurationError("Number of keys and input files should match in join")
        inputs = [self._read_jsonl(file_open(fname)) for fname in infiles]
        self._write_jsonl(_gen(inputs, keys), outfile)
