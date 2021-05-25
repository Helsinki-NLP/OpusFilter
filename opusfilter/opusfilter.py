"""Processor for filter configurations"""

import collections
import copy
import functools
import itertools
import logging
import operator
import os
import pickle
import random

import json
import numpy as np
from tqdm import tqdm

from opustools import OpusRead

from . import ConfigurationError
from . import pipeline
from . import lm
from . import word_alignment
from . import tokenization
from . import classifier
from . import segment_hash
from .util import file_open

logger = logging.getLogger(__name__)


def dict_get(key, dictionary):
    """Recursive get for multi-part key with dot (.) as separator

    Example:
    dict_get("foo.bar", {"foo": {"bar": 5}}) -> 5

    Raises KeyError for missing keys.

    """
    parts = key.split('.')
    first = parts.pop(0)
    if isinstance(dictionary, list):
        value = dictionary[int(first)]
    else:
        value = dictionary[first]
    return value if not len(parts) else dict_get('.'.join(parts), value)


def dict_set(key, value, dictionary):
    """Recursive set for multi-part key with dot (.) as separator

    Example:
    dict_set("foo.x", 1, {"foo": {"bar": 5}}) -> {"foo": {"bar": 5, "x": 1}}

    Creates new sub-dictionaries if needed. However, if a key exists
    with a non-dictionary value, TypeError is raised.

    """
    parts = key.split('.')
    while parts:
        first = parts.pop(0)
        if not parts:
            dictionary[first] = value
            return
        if first not in dictionary:
            dictionary[first] = {}
        dictionary = dictionary[first]


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
            'train_classifier': self.train_classifier,
            'classify': self.classify,
            'join': self.join_scores,
            'sort': self.sort_files,
            'head': self.head,
            'tail': self.tail,
            'product': self.product,
            'remove_duplicates': self.remove_duplicates,
            'split': self.split,
            'unzip': self.unzip,
            'preprocess': self.preprocess
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
        if 'release' not in parameters:
            logger.info("No release version provided for corpus %s, using 'latest'",
                        parameters['corpus_name'])
            parameters['release'] = 'latest'
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
    def pair_generator(*filenames, tokenizers=None):
        """Yield and optionally tokenize sentence pairs from given files"""
        if tokenizers is None:
            tokenizers = [None] * len(filenames)
        tokenize_funcs = [tokenization.get_tokenize(tokenizer) for tokenizer in tokenizers]
        files = [file_open(fname) for fname in filenames]
        lines = [f.readline() for f in files]
        while all(lines):
            yield tuple(tokenize(line.rstrip()) for tokenize, line in zip(tokenize_funcs, lines))
            lines = [f.readline() for f in files]
        for fobj in files:
            fobj.close()

    def fix_filter_file_paths(self, filter_params):
        """Fix file paths in filter parameters"""
        # Make a copy so that the original paths are not modified
        fixed_params = copy.deepcopy(filter_params)
        for f in fixed_params:
            filter_name = next(iter(f.items()))[0]
            if filter_name == 'WordAlignFilter' and 'priors' in f[filter_name]:
                f[filter_name]['priors'] = os.path.join(
                    self.output_dir, f[filter_name]['priors'])
            elif filter_name in 'CrossEntropyFilter':
                for idx, lm_params in enumerate(f[filter_name]['lm_params']):
                    f[filter_name]['lm_params'][idx]['filename'] = os.path.join(
                        self.output_dir, lm_params['filename'])
                    if lm_params.get('interpolate'):
                        for idx2 in range(len(lm_params['interpolate'])):
                            f[filter_name]['lm_params'][idx]['interpolate'][idx2][0] = os.path.join(
                                self.output_dir, lm_params['interpolate'][idx2][0])
            elif filter_name in 'CrossEntropyDifferenceFilter':
                for key in ['id_lm_params', 'nd_lm_params']:
                    for idx, lm_params in enumerate(f[filter_name][key]):
                        f[filter_name][key][idx]['filename'] = os.path.join(
                            self.output_dir, lm_params['filename'])
                        if lm_params.get('interpolate'):
                            for idx2 in range(len(lm_params['interpolate'])):
                                f[filter_name][key][idx]['interpolate'][idx2][0] = os.path.join(
                                    self.output_dir, lm_params['interpolate'][idx2][0])
        return fixed_params

    def filter_data(self, parameters, overwrite=False):
        """Write sentences to file if they pass given filters"""
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in sort")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        filter_params = self.fix_filter_file_paths(parameters['filters'])
        filter_pipe = pipeline.FilterPipeline.from_config(filter_params)
        filterfalse = parameters.get('filterfalse', False)
        pairs_gen = self.pair_generator(*infiles)
        if filterfalse:
            pairs = filter_pipe.filterfalse(pairs_gen)
        else:
            pairs = filter_pipe.filter(pairs_gen)
        limit = parameters.get('limit')
        outfileobjs = [file_open(fname, 'w') for fname in outfiles]
        for idx, pair in tqdm(enumerate(pairs)):
            for item, fobj in zip(pair, outfileobjs):
                fobj.write(item+'\n')
                fobj.flush()
            if limit and idx >= limit - 1:
                break
        for fobj in outfileobjs:
            fobj.close()

    def concatenate(self, parameters, overwrite=False):
        """Concatenate files"""
        outfile = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(outfile):
            logger.info("Output file exists, skipping step")
            return
        with file_open(outfile, 'w') as outf:
            for infile in parameters['inputs']:
                logger.info("opening %s", os.path.join(self.output_dir, infile))
                with file_open(os.path.join(self.output_dir, infile)) as inf:
                    for line in inf:
                        outf.write(line.rstrip() + '\n')

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

        Keeps the order of lines, unless if shuffle_subset is True in
        parameters, in which case the target lines will be in a random
        order.

        """
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in sort")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        random.seed(parameters.get('seed', None))
        size = parameters['size']
        shuffle_subset = parameters.get('shuffle_subset', False)
        total = self._get_total_lines(infiles[0])
        logger.info("Sampling subset of %s lines from total %s lines", size, total)
        if shuffle_subset:
            sample = random.sample(range(total), size)
            with file_open(infiles[0]) as inf, file_open(outfiles[0], 'w') as outf:
                for line in self._yield_subset(inf, sample):
                    outf.write(line)
            for infname, outfname in zip(infiles[1:], outfiles[1:]):
                with file_open(infname) as inf:
                    lines = [line for line in self._yield_subset(inf, sample)]
                random.shuffle(lines)
                with file_open(outfname, 'w') as outf:
                    for line in lines:
                        outf.write(line)
        else:
            sample = random.sample(range(total), size)
            for infname, outfname in zip(infiles, outfiles):
                with file_open(infname) as inf, \
                     file_open(outfname, 'w') as outf:
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
            tokenizers=[parameters['parameters'].get('src_tokenizer', None),
                        parameters['parameters'].get('tgt_tokenizer', None)]))
        word_alignment.make_priors(
            pair_gen, model_out, model=parameters['parameters'].get('model', 3))

    @staticmethod
    def _write_jsonl(objects, fname):
        """Write objects to file as JSON lines"""
        with file_open(fname, 'w') as fobj:
            for obj in objects:
                fobj.write(json.dumps(obj, sort_keys=True)+'\n')

    @staticmethod
    def _read_jsonl(fname):
        """Return a generator for items in JSON lines file"""
        with file_open(fname, 'r') as fobj:
            for line in fobj:
                yield json.loads(line)

    def score_data(self, parameters, overwrite=False):
        """Score language data based on given filters"""
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        score_out = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(score_out):
            logger.info("Output file exists, skipping step")
            return
        filter_params = self.fix_filter_file_paths(parameters['filters'])
        filter_pipe = pipeline.FilterPipeline.from_config(filter_params)
        scores_gen = filter_pipe.score(self.pair_generator(*infiles))
        self._write_jsonl(scores_gen, score_out)

    def train_classifier(self, parameters, overwrite=False):
        """Train classifier for scored sentence pairs"""
        model_out = os.path.join(self.output_dir, parameters['model'])
        if not overwrite and os.path.isfile(model_out):
            logger.info("Output file exists, skipping step")
            return
        training_scores = os.path.join(self.output_dir, parameters['training_scores'])
        dev_scores = os.path.join(self.output_dir, parameters['dev_scores']) \
            if 'dev_scores' in parameters else None
        trainer = classifier.TrainClassifier(
            training_scores=training_scores,
            dev_scores=dev_scores, model_type=parameters['model_type'],
            model_parameters=parameters['model_parameters'],
            features=parameters['features']
        )
        model, value, features = trainer.find_best_model(
            parameters['criterion'], **parameters.get('optimization', {}))

        logger.info('Best model has {criterion}: {value}'.format(
            criterion=parameters['criterion'], value=value))

        feature_cutoffs = ''
        for item in features.items():
            feature_cutoffs += '\n\t'+str(item)
        logger.info('And feature cutoffs: {}'.format(feature_cutoffs))

        feature_weights = ''
        for item in model.weights():
            feature_weights += '\n\t'+str(item)
        logger.info('And weights: {}'.format(feature_weights))

        logger.info('Saving best model to {}'.format(model_out))

        # with file_open(model_out, 'wb') as model_file:
        # TODO: ValueError: binary mode doesn't take an encoding argument
        with open(model_out, 'wb') as model_file:
            pickle.dump(model, model_file)

    def classify(self, parameters, overwrite=False):
        """Assign classifier probabilities and/or labels to scored sentence pairs"""
        labels_out = os.path.join(
            self.output_dir, parameters['output_labels']) \
            if 'output_labels' in parameters else None
        probs_out = os.path.join(
            self.output_dir, parameters['output_probabilities']) \
            if 'output_probabilities' in parameters else None
        if (not overwrite and
                (labels_out is None or os.path.isfile(labels_out)) and
                (probs_out is None or os.path.isfile(probs_out))):
            logger.info("Output files exists, skipping step")
            return
        model_in = os.path.join(self.output_dir, parameters['model'])
        # with file_open(model_in, 'rb') as model_file:
        # TODO: ValueError: binary mode doesn't take an encoding argument
        with open(model_in, 'rb') as model_file:
            model = pickle.load(model_file)
        scores_in = os.path.join(self.output_dir, parameters['scores'])
        true_label = parameters.get('true_label', None)
        chunksize = parameters.get('chunksize', 100000)
        if labels_out:
            model.write_preds(scores_in, labels_out, true_label, chunksize=chunksize)
        if probs_out:
            model.write_probs(scores_in, probs_out, true_label, chunksize=chunksize)

    @staticmethod
    def _read_values(fobj, key=None, conv=None, combine=None):
        """Return a generator for values in score file

        The file should contain one JSON object per line. If the line
        cannot be interpreted as a JSON object, it is taken as a
        string. If conv is not None, conv(value) is yielded instead of
        the plain value. Values of multiple keys are combined with the
        given operator from the operator module, or returned as a list
        if combine is None.

        """
        if combine and not hasattr(operator, combine):
            raise ConfigurationError(
                "Combine operator {} not found in the operator module".format(combine))
        for line in fobj:
            try:
                val = json.loads(line)
            except json.decoder.JSONDecodeError:
                val = line
            if isinstance(key, str):
                val = dict_get(key, val)
                if conv is not None:
                    val = conv(val)
            elif isinstance(key, (list, tuple)):
                val = [dict_get(k, val) for k in key]
                if conv is not None:
                    val = [conv(v) for v in val]
                if combine:
                    oper = getattr(operator, combine)
                    val = functools.reduce(oper, val)
            yield val

    def sort_files(self, parameters, overwrite=False):
        """Sort file(s) by values read from other file"""
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in sort")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        valuefile = os.path.join(self.output_dir, parameters['values'])
        reverse = parameters.get('reverse', False)
        key = parameters.get('key')
        typeconv = parameters.get('type')
        if typeconv is not None:
            typeconv = {'float': float, 'int': int, 'str': str}[typeconv]
        combine = parameters.get('combine_operator')
        with file_open(valuefile, 'r') as fobj:
            logger.info("Reading values from %s", valuefile)
            values = [x for x in tqdm(
                self._read_values(fobj, key=key, conv=typeconv, combine=combine))]
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
                        dict_set(keys[idx], obj, new)
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
        inputs = [self._read_jsonl(fname) for fname in infiles]
        self._write_jsonl(_gen(inputs, keys), outfile)

    def slice(self, parameters, overwrite=False):
        """Take slice from file(s)"""
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in head")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        start = parameters.get('start', 0)
        stop = parameters.get('stop')
        step = parameters.get('step', 1)
        for infile, outfile in zip(infiles, outfiles):
            logger.info("Processing file %s", infile)
            with file_open(infile, 'r') as inf, file_open(outfile, 'w') as outf:
                for line in tqdm(itertools.islice(inf, start, stop, step)):
                    outf.write(line)

    def head(self, parameters, overwrite=False):
        """Take the first n lines from file(s)"""
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in head")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        n = parameters['n']
        for infile, outfile in zip(infiles, outfiles):
            logger.info("Processing file %s", infile)
            with file_open(infile, 'r') as inf, file_open(outfile, 'w') as outf:
                for line in tqdm(itertools.islice(inf, n)):
                    outf.write(line)

    def tail(self, parameters, overwrite=False):
        """Take the last n lines from file(s)"""
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in head")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        n = parameters['n']
        for infile, outfile in zip(infiles, outfiles):
            logger.info("Processing file %s", infile)
            with file_open(infile, 'r') as inf, file_open(outfile, 'w') as outf:
                tmp = []
                for line in tqdm(inf):
                    tmp.append(line)
                    if len(tmp) > n:
                        tmp.pop(0)
                for line in tmp:
                    outf.write(line)

    @staticmethod
    def _multipair_gen(lists_of_files):
        """Generator for lines in lists of parallel files"""
        infs = [[file_open(infile) for infile in infiles] for infiles in lists_of_files]
        lines = [[fobj.readline() for fobj in flist] for flist in infs]
        while all(line for linelist in lines for line in linelist):
            yield lines
            lines = [[fobj.readline() for fobj in flist] for flist in infs]
        for folist in infs:
            for fobj in folist:
                fobj.close()

    def product(self, parameters, overwrite=False):
        """Sample a product of segments from lists of alternative files"""
        infilelists = [
            [os.path.join(self.output_dir, fname) for fname in filelist]
            for filelist in parameters['inputs']
        ]
        outfiles = [
            os.path.join(self.output_dir, fname) for fname in parameters['outputs']
        ]
        if len(outfiles) != len(infilelists):
            raise ConfigurationError(
                "Number of input file lists and output files should match in product")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        skip_empty = parameters.get('skip_empty', True)
        skip_duplicates = parameters.get('skip_duplicates', True)
        sample_k = parameters.get('k', None)
        random.seed(parameters.get('seed', None))
        outfs = [file_open(outfile, 'w') for outfile in outfiles]
        for lines in tqdm(self._multipair_gen(infilelists)):
            if skip_empty:
                lines = [
                    [line for line in linelist if line.strip()]
                    for linelist in lines
                ]
            if skip_duplicates:
                lines = [sorted(set(linelist)) for linelist in lines]
            product = list(itertools.product(*lines))
            if sample_k is not None:
                random.shuffle(product)
                product = product[:sample_k]
            for pair in product:
                for fobj, line in zip(outfs, pair):
                    fobj.write(line)

    def split(self, parameters, overwrite=False):
        """Split parallel files to two subsets"""
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        outfiles_2 = [os.path.join(self.output_dir, fname) for fname in parameters['outputs_2']] \
            if 'outputs_2' in parameters else []
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles) or (outfiles_2 and len(outfiles_2) != len(infiles)):
            raise ConfigurationError(
                "Number of input and output files should match in split")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles + outfiles_2):
            logger.info("Output files exists, skipping step")
            return
        divisor = parameters['divisor']
        threshold = parameters.get('threshold', 1)
        hasher = segment_hash.SegmentHasher(
            compare=parameters.get('compare', 'all'),
            hash=parameters.get('hash', 'xx_64'),
            hashseed=parameters.get('seed', 0)
        )
        infs = [file_open(infile) for infile in infiles]
        outfs = [file_open(outfile, 'w') for outfile in outfiles]
        outfs_2 = [file_open(outfile, 'w') for outfile in outfiles_2]
        hits = 0
        total = 0
        for lines in tqdm(zip(*infs)):
            total += 1
            key = hasher.apply(lines)
            if key % divisor < threshold:
                hits += 1
                for idx, line in enumerate(lines):
                    outfs[idx].write(line)
            elif outfs_2:
                for idx, line in enumerate(lines):
                    outfs_2[idx].write(line)
        logger.info(
            "Split {} lines to {} ({:.2f}%) and {} ({:.2f}%) lines".format(
                total, hits, 100 * hits / total, total - hits, 100 * (total - hits) / total))
        for idx in range(len(infiles)):
            infs[idx].close()
            outfs[idx].close()
            if outfs_2:
                outfs_2[idx].close()

    def remove_duplicates(self, parameters, overwrite=False):
        """Remove duplicates from parallel lines in files"""
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError(
                "Number of input and output files should match in remove_duplicates")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        hasher = segment_hash.SegmentHasher(
            compare=parameters.get('compare', 'all'),
            hash=parameters.get('hash', 'xx_64'),
            letters_only=parameters.get('letters_only', False),
            lowercase=parameters.get('lowercase', False),
        )
        infs = [file_open(infile) for infile in infiles]
        outfs = [file_open(outfile, 'w') for outfile in outfiles]
        overlap = parameters.get('overlap', None)
        if overlap:
            overlapfs = [file_open(infile) for infile in overlap]
            overlap_total = 0
            overlap_counter = collections.Counter()
            for lines in tqdm(zip(*overlapfs)):
                key = hasher.apply(lines)
                overlap_total += 1
                overlap_counter[key] += 1
            logger.info(
                "Collected {} types from {} tokens".format(
                    len(overlap_counter), overlap_total))
        counter = collections.Counter()
        removed_entries = 0
        total = 0
        for lines in tqdm(zip(*infs)):
            total += 1
            key = hasher.apply(lines)
            if overlap:
                if key in overlap_counter:
                    counter[key] += 1
                    removed_entries += 1
                    continue
            else:
                counter[key] += 1
                if counter[key] > 1:
                    removed_entries += 1
                    continue
            for idx, line in enumerate(lines):
                outfs[idx].write(line)
        removed_types = sum(1 for c in counter.values() if c > 1)
        logger.info(
            "Removed {} / {} = {:.2f}% duplicate lines (duplicate types: {})".format(
                removed_entries, total, 100 * removed_entries / total if total > 0 else 0, removed_types))
        for idx in range(len(infiles)):
            infs[idx].close()
            outfs[idx].close()

    def unzip(self, parameters, overwrite=False):
        """Unzip parallel segments joined in a single file into multiple files"""
        infile = os.path.join(self.output_dir, parameters['input'])
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        separator = parameters['separator']
        outfs = [file_open(outfile, 'w') for outfile in outfiles]
        with file_open(infile, 'r') as inf:
            for idx, line in tqdm(enumerate(inf)):
                parts = line.split(separator)
                if len(parts) != len(outfiles):
                    raise ConfigurationError(
                        "Number output files do not match the %s parts in line %s" % (len(parts), idx))
                for part, outf in zip(parts, outfs):
                    outf.write(part.strip() + '\n')

    def preprocess(self, parameters, overwrite=False):
        """Run preprocessors on text data"""
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in preprocess")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        preprocess_pipe = pipeline.PreprocessorPipeline.from_config(parameters['preprocessors'])
        pairs = preprocess_pipe.process(self.pair_generator(*infiles))
        outfileobjs = [file_open(fname, 'w') for fname in outfiles]
        for pair in tqdm(pairs):
            for item, fobj in zip(pair, outfileobjs):
                fobj.write(item + '\n')
                fobj.flush()
        for fobj in outfileobjs:
            fobj.close()
