"""Processor for filter configurations"""

import collections
import copy
import functools
import itertools
import logging
import math
import multiprocessing
import operator
import os
import pickle
import random
import tempfile
from itertools import chain

import json
from tqdm import tqdm

from . import ConfigurationError, OpusFilterRuntimeError
from . import embeddings
from . import lm
from . import pipeline
from . import subwords
from . import segment_hash
from . import tokenization
from . import word_alignment
from .util import file_open, file_download, Var, VarStr, count_lines


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
    return value if not parts else dict_get('.'.join(parts), value)


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


class ParallelWrapper:
    """Decorator for parallelizing OpusFilter steps

    This decorator will split "inputs" and "outputs" (or "output")
    into shareds and process them in parallel. Finally, all the
    intermediate result will be merged into a single file. All the
    intermediate files will be deleted.

    """

    def __init__(self, extra_parameters):
        if "inputs" not in extra_parameters:
            raise ConfigurationError("'inputs' is required in extra_parameters to parallelize.")
        if "outputs" not in extra_parameters and "output" not in extra_parameters:
            raise ConfigurationError("'outputs' or 'output' is required in extra_parameters to parallelize.")
        self.extra_parameters = extra_parameters
        self.func = None

    def __call__(self, func):
        self.func = func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.parallelize(*args, **kwargs)
        return wrapper

    @staticmethod
    def split(infiles, outfiles, n_jobs):
        """split files into parts, and write them to temporary files"""
        chunk_size = int(math.ceil(count_lines(infiles[0]) / n_jobs))
        in_chunked_files = []
        out_chunked_files = []
        infileobjs = [file_open(infile) for infile in infiles]
        for i, lines in enumerate(zip(*infileobjs)):
            if i % chunk_size == 0:
                intmpfiles = [tempfile.mkstemp(dir=os.path.dirname(infile),
                              suffix=f".part{str(i // chunk_size)}.{os.path.basename(infile)}")[1] for infile in infiles]
                in_chunked_files.append(intmpfiles)
                intmpfiles_objs = [file_open(intmpfile, mode="w") for intmpfile in intmpfiles]
                outtmpfiles = [tempfile.mktemp(dir=os.path.dirname(outfile),
                               suffix=f".part{str(i // chunk_size)}.{os.path.basename(outfile)}") for outfile in outfiles]
                out_chunked_files.append(outtmpfiles)

            for line, tmpfile in zip(lines, intmpfiles_objs):
                tmpfile.write(line)
        return in_chunked_files, out_chunked_files

    @staticmethod
    def merge(in_chunked_files, outfiles, out_chunked_files, limit):
        """merge temporary files into final files and delete temporary files"""
        for outfile, parts in zip(outfiles, zip(*out_chunked_files)):
            with file_open(outfile, 'w') as out:
                finput = chain.from_iterable(file_open(part) for part in parts)
                for i, line in enumerate(finput):
                    out.write(line)
                    if limit and i >= limit - 1:
                        break
            for part in parts:
                os.unlink(part)
        for parts in in_chunked_files:
            for part in parts:
                os.unlink(part)

    def parallelize(self, obj, parameters, overwrite=False):
        """Wrapper for parallelizing a function"""
        # check if parameters are valid
        n_jobs = parameters.pop('n_jobs', obj.default_n_jobs)
        if n_jobs <= 1:
            self.func(obj, parameters, overwrite)
            return
        # pylint: disable=W0212
        obj._check_extra_parameters(self.extra_parameters, parameters)
        infiles = [os.path.join(obj.output_dir, fname) for fname in parameters['inputs']]
        if "outputs" in parameters:
            outfiles = [os.path.join(obj.output_dir, fname) for fname in parameters['outputs']]
        elif "output" in parameters:  # function `score` use `output` instead of `outputs`
            outfiles = [os.path.join(obj.output_dir, parameters['output'])]
        if len(outfiles) != len(infiles) and "outputs" in parameters:
            raise ConfigurationError("Number of input and output files should match in sort")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        in_chunked_files, out_chunked_files = self.split(infiles, outfiles, n_jobs)
        # run jobs in parallel
        sub_processes = []
        for intmpfiles, outtmpfiles in zip(in_chunked_files, out_chunked_files):
            parameters_i = copy.deepcopy(parameters)
            parameters_i["inputs"] = intmpfiles
            if "outputs" in parameters:
                parameters_i["outputs"] = [os.path.relpath(path, obj.output_dir) for path in outtmpfiles]
            elif "output" in parameters:  # function `score` use `output` instead of `outputs`
                parameters_i["output"] = os.path.relpath(outtmpfiles[0], obj.output_dir)
            process = multiprocessing.Process(target=self.func, args=(obj, parameters_i, overwrite))
            process.daemon = True
            process.start()
            sub_processes.append(process)
        for process in sub_processes:
            process.join()
        limit = parameters.get('limit', None)
        self.merge(in_chunked_files, outfiles, out_chunked_files, limit)


# pylint: disable=R0904
class OpusFilter:
    """Apply filters to language data"""

    def __init__(self, configuration):
        self.configuration = configuration
        self.output_dir = configuration.get('common', {}).get('output_directory')
        if not self.output_dir:
            logger.warning('Output directory not specified. Writing files to current directory.')
            self.output_dir = '.'
        elif not os.path.isdir(self.output_dir):
            logger.warning('Directory "%s" does not exist. It will be created.', self.output_dir)
            os.mkdir(self.output_dir)
        self.constants = configuration.get('common', {}).get('constants', {})
        self.chunksize = configuration.get('common', {}).get('chunksize', 100000)
        self.default_n_jobs = configuration.get('common', {}).get('default_n_jobs', 1)
        self.step_functions = {
            'opus_read': self.read_from_opus,
            'filter': self.filter_data,
            'concatenate': self.concatenate,
            'subset': self.get_subset,
            'train_bpe': self.train_bpe,
            'train_morfessor': self.train_morfessor,
            'train_ngram': self.train_ngram,
            'train_alignment': self.train_alignment,
            'train_nearest_neighbors': self.train_nearest_neighbors,
            'score': self.score_data,
            'train_classifier': self.train_classifier,
            'classify': self.classify,
            'join': self.join_scores,
            'sort': self.sort_files,
            'head': self.head,
            'tail': self.tail,
            'slice': self.slice,
            'product': self.product,
            'remove_duplicates': self.remove_duplicates,
            'split': self.split,
            'unzip': self.unzip,
            'preprocess': self.preprocess,
            'download': self.download_file,
            'write': self.write_to_file
        }

    def execute_steps(self, overwrite=False, last=None):
        """Execute steps in the same order as they are in the configuration"""
        for num, step in enumerate(self.configuration['steps']):
            if last is not None and num + 1 > last:
                logger.info('Stopping after step %s', last)
                break
            self._run_step(step, num + 1, overwrite)

    def execute_step(self, num, overwrite=False):
        """Execute single step in the configuration (first = 1, last = -1)

        Does not check any dependencies and may fail if the input
        files do not exist.

        """
        step = self.configuration['steps'][num if num < 0 else num - 1]
        self._run_step(step, num, overwrite)

    @staticmethod
    def _check_variables(variables):
        """Check that variable definitions are valid"""
        lengths = set()
        for key, value in variables.items():
            if not isinstance(value, list):
                raise ConfigurationError(f"Variable {key} does not define a list")
            lengths.add(len(value))
            if len(lengths) > 1:
                raise ConfigurationError(
                    f"Variable {key} has a different length of values than the previous")
        return list(lengths)[0] if lengths else 0

    def _expand_parameters(self, obj, namespace):
        """Expand Var and VarStr objects in obj"""
        if isinstance(obj, list):
            return [self._expand_parameters(x, namespace) for x in obj]
        if isinstance(obj, dict):
            return {self._expand_parameters(key, namespace): self._expand_parameters(value, namespace)
                    for key, value in obj.items()}
        if isinstance(obj, VarStr):
            try:
                formatted = obj.value.format(**namespace)
            except (KeyError, IndexError) as err:
                raise ConfigurationError(
                    f"String substitutions not defined in the context: {obj.value}") from err
            return formatted
        if isinstance(obj, Var):
            if obj.value not in namespace:
                raise ConfigurationError(f"Variable not defined in the context: {obj.value}")
            return namespace[obj.value]
        return obj

    def _run_step(self, step, num, overwrite):
        """Run given step"""
        logger.info('Running step %s: %s', num, step['type'])
        variables = step.get('variables', {})
        namespace = copy.copy(self.constants)
        namespace.update(step.get('constants', {}))
        if variables:
            num_choices = self._check_variables(variables)
            if not num_choices:
                logger.warning("Variable value lists are empty, skipping step")
            for idx in range(num_choices):
                for key, values in variables.items():
                    namespace[key] = values[idx]
                parameters = self._expand_parameters(step['parameters'], namespace)
                logger.info("- substep %s: %s", idx + 1, dict(namespace))
                logger.debug("  parameters: %s", parameters)
                self.step_functions[step['type']](parameters, overwrite=overwrite)
        else:
            parameters = self._expand_parameters(step['parameters'], namespace)
            logger.debug("  parameters: %s", parameters)
            self.step_functions[step['type']](parameters, overwrite=overwrite)

    @staticmethod
    def _check_extra_parameters(valid_keys, parameters):
        """Warn if parameters dict has keys outside valid_keys"""
        extra = []
        for key in parameters:
            if key not in valid_keys:
                logger.error("Unknown parameter '%s' with value: %s", key, parameters[key])
                extra.append(key)
        if extra:
            raise ConfigurationError(f"Unknown parameter keys: {', '.join(extra)}")

    def read_from_opus(self, parameters, overwrite=False):
        """Download and read a corpus from OPUS using OpusTools

        For details, see:
        * OPUS corpus collection :cite:`tiedemann-2016-parallel`.
        * OpusTools :cite:`aulamo-etal-2020-opustools`.

        """
        from opustools import OpusRead
        self._check_extra_parameters(
            {'src_output', 'tgt_output', 'suppress_prompts', 'release', 'corpus_name',
             'source_language', 'target_language', 'preprocessing'}, parameters)
        src_out = os.path.join(self.output_dir, parameters['src_output'])
        tgt_out = os.path.join(self.output_dir, parameters['tgt_output'])
        if not overwrite and os.path.isfile(src_out) and os.path.isfile(tgt_out):
            logger.info("Output files exists, skipping step")
            return
        if 'suppress_prompts' not in parameters:
            logger.warning("By default prompting to download corpus")
            logger.warning("To suppress prompts include `suppress_prompts: true`")
            parameters['suppress_prompts'] = False
        if 'release' not in parameters:
            logger.info("No release version provided for corpus %s, using 'latest'",
                        parameters['corpus_name'])
            parameters['release'] = 'latest'
        opus_reader = OpusRead(
            directory=parameters['corpus_name'],
            source=parameters['source_language'],
            target=parameters['target_language'],
            release=parameters['release'],
            suppress_prompts=parameters['suppress_prompts'],
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

    @staticmethod
    def _close_files(*files):
        """Close multiple file objects"""
        for fobj in files:
            fobj.close()

    @ParallelWrapper({'inputs', 'outputs', 'filters', 'filterfalse', 'limit'})
    def filter_data(self, parameters, overwrite=False):
        """Write sentences to file if they pass given filters"""
        # no need to check extra parameters, they are checked in the parallel wrapper
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in sort")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        filter_pipe = pipeline.FilterPipeline.from_config(parameters['filters'], workdir=self.output_dir)
        filter_pipe.chunksize = self.chunksize
        pairs_gen = self.pair_generator(*infiles)
        if parameters.get('filterfalse', False):
            pairs = filter_pipe.filterfalse(pairs_gen)
        else:
            pairs = filter_pipe.filter(tqdm(pairs_gen))
        limit = parameters.get('limit')
        outfileobjs = [file_open(fname, 'w') for fname in outfiles]
        for idx, pair in enumerate(pairs):
            for item, fobj in zip(pair, outfileobjs):
                fobj.write(item+'\n')
                fobj.flush()
            if limit and idx >= limit - 1:
                break
        self._close_files(*outfileobjs)

    def concatenate(self, parameters, overwrite=False):
        """Concatenate files"""
        self._check_extra_parameters({'inputs', 'output'}, parameters)
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
        self._check_extra_parameters({'inputs', 'outputs', 'seed', 'size', 'shuffle_subset'}, parameters)
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
                    lines = list(self._yield_subset(inf, sample))
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

    def train_bpe(self, parameters, overwrite=False):
        """Train morphological segmentation using Byte-Pair Encoding (BPE)"""
        self._check_extra_parameters({'model', 'input', 'symbols', 'min_frequency', 'num_workers'}, parameters)
        model_out = os.path.join(self.output_dir, parameters['model'])
        if not overwrite and os.path.isfile(model_out):
            logger.info("Output file exists, skipping step")
            return
        subwords.BPESegmentation.train(
            os.path.join(self.output_dir, parameters['input']), model_out, parameters.get('symbols', 10000),
            min_frequency=parameters.get('min_frequency', 2), num_workers=parameters.get('num_workers', 1))

    def train_morfessor(self, parameters, overwrite=False):
        """Train morphological segmentation using Morfessor"""
        self._check_extra_parameters({'model', 'input', 'corpusweight', 'min_frequency', 'dampening', 'seed',
                                      'use_skips', 'forcesplit_list', 'nosplit_re'}, parameters)
        model_out = os.path.join(self.output_dir, parameters['model'])
        if not overwrite and os.path.isfile(model_out):
            logger.info("Output file exists, skipping step")
            return
        subwords.MorfessorSegmentation.train(
            os.path.join(self.output_dir, parameters['input']), model_out,
            corpusweight=parameters.get('corpusweight', 1.0),
            min_frequency=parameters.get('min_frequency', 1),
            dampening=parameters.get('dampening', 'log'),
            seed=parameters.get('seed'),
            use_skips=parameters.get('use_skips', True),
            forcesplit_list=parameters.get('forcesplit_list'),
            nosplit_re=parameters.get('nosplit_re'))

    def train_ngram(self, parameters, overwrite=False):
        """Train an n-gram language model"""
        self._check_extra_parameters({'model', 'data', 'parameters'}, parameters)
        model_out = os.path.join(self.output_dir, parameters['model'])
        if not overwrite and os.path.isfile(model_out):
            logger.info("Output file exists, skipping step")
            return
        data_name = parameters['data']
        lmtoken_params = lm.join_workdir_to_lm_paths(parameters['parameters'], self.output_dir)
        tokenizer = lm.LMTokenizer(**lmtoken_params)
        with tempfile.NamedTemporaryFile('w+b', suffix='.seg.gz') as segfile:
            num = 0
            with file_open(os.path.join(self.output_dir, data_name), 'r') as infile, \
                 file_open(os.path.join(self.output_dir, segfile.name), 'w') as outfile:
                for num, line in enumerate(tqdm(infile)):
                    tokens = tokenizer.tokenize(line.strip())
                    outfile.write(' '.join(tokens) + '\n')
            if num == 0:
                raise OpusFilterRuntimeError(f"No training data available in {data_name}")
            lm.train(os.path.join(self.output_dir, segfile.name), model_out,
                     **parameters['parameters'])

    def train_alignment(self, parameters, overwrite=False):
        """Train eflomal alignment priors"""
        self._check_extra_parameters({'src_data', 'tgt_data', 'scores', 'parameters', 'output'}, parameters)
        self._check_extra_parameters({'src_tokenizer', 'tgt_tokenizer', 'model'}, parameters.get('parameters'))
        model_out = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(model_out):
            logger.info("Output file exists, skipping step")
            return
        score_file = os.path.join(self.output_dir, parameters['scores']) if 'scores' in parameters else None
        pair_gen = tqdm(self.pair_generator(
            os.path.join(self.output_dir, parameters['src_data']),
            os.path.join(self.output_dir, parameters['tgt_data']),
            tokenizers=[parameters['parameters'].get('src_tokenizer', None),
                        parameters['parameters'].get('tgt_tokenizer', None)]))
        word_alignment.make_priors(
            pair_gen, model_out, model=parameters['parameters'].get('model', 3), score_file=score_file)

    def train_nearest_neighbors(self, parameters, overwrite=False):
        """Train model for querying nearest neighbors"""
        model_out = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(model_out):
            logger.info("Output file exists, skipping step")
            return
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        model = embeddings.ParallelNearestNeighbors(
            input_files=infiles, languages=parameters['languages'],
            n_neighbors=parameters.get('n_neighbors', 4), algorithm=parameters.get('algorithm', 'brute'),
            metric=parameters.get('metric', 'cosine'))
        with open(model_out, 'wb') as model_file:
            pickle.dump(model, model_file)

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

    @ParallelWrapper({'inputs', 'output', "filters"})
    def score_data(self, parameters, overwrite=False):
        """Score language data based on given filters"""
        # no need to check extra parameters, they are checked in the parallel wrapper
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        score_out = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(score_out):
            logger.info("Output file exists, skipping step")
            return
        filter_pipe = pipeline.FilterPipeline.from_config(parameters['filters'], workdir=self.output_dir)
        filter_pipe.chunksize = self.chunksize
        scores_gen = filter_pipe.score(self.pair_generator(*infiles))
        self._write_jsonl(scores_gen, score_out)

    def train_classifier(self, parameters, overwrite=False):
        """Train classifier for scored sentence pairs"""
        from . import classifier
        self._check_extra_parameters(
            {'model', 'training_scores', 'dev_scores', 'model_type', 'model_parameters',
             'features', 'optimization', 'criterion'}, parameters)
        model_out = os.path.join(self.output_dir, parameters['model'])
        if not overwrite and os.path.isfile(model_out):
            logger.info("Output file exists, skipping step")
            return
        training_scores = os.path.join(self.output_dir, parameters['training_scores'])
        dev_scores = os.path.join(self.output_dir, parameters['dev_scores']) \
            if 'dev_scores' in parameters else None
        trainer = classifier.TrainClassifier(
            training_scores=training_scores,
            dev_scores=dev_scores, model_type=parameters.get('model_type'),
            model_parameters=parameters.get('model_parameters'),
            features=parameters['features']
        )
        model, value, features = trainer.find_best_model(
            parameters['criterion'], **parameters.get('optimization', {}))

        logger.info('Best model has %s: %s', parameters['criterion'], value)

        feature_cutoffs = ''
        for item in features.items():
            feature_cutoffs += '\n\t'+str(item)
        logger.info('And feature cutoffs: %s', feature_cutoffs)

        feature_weights = ''
        for item in model.weights():
            feature_weights += '\n\t'+str(item)
        logger.info('And weights: %s', feature_weights)

        logger.info('Saving best model to %s', model_out)
        with open(model_out, 'wb') as model_file:
            pickle.dump(model, model_file)

    def classify(self, parameters, overwrite=False):
        """Assign classifier probabilities and/or labels to scored sentence pairs"""
        self._check_extra_parameters(
            {'scores', 'model', 'output_labels', 'output_probabilities', 'true_label', 'chunksize'}, parameters)
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
            raise ConfigurationError(f"Combine operator {combine} not found in the operator module")
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
        import numpy as np
        self._check_extra_parameters(
            {'inputs', 'outputs', 'key', 'type', 'values', 'combine_operator', 'reverse'}, parameters)
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in sort")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        valuefile = os.path.join(self.output_dir, parameters['values'])
        typeconv = parameters.get('type')
        if typeconv is not None:
            typeconv = {'float': float, 'int': int, 'str': str}[typeconv]
        with file_open(valuefile, 'r') as fobj:
            logger.info("Reading values from %s", valuefile)
            values = list(tqdm(self._read_values(
                fobj, key=parameters.get('key'), conv=typeconv, combine=parameters.get('combine_operator'))))
            order = list(np.argsort(values))
            if parameters.get('reverse', False):
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

        self._check_extra_parameters({'inputs', 'output', 'keys'}, parameters)
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
        self._check_extra_parameters({'inputs', 'outputs', 'start', 'stop', 'step'}, parameters)
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
        self._check_extra_parameters({'inputs', 'outputs', 'n'}, parameters)
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in head")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        num = parameters['n']
        for infile, outfile in zip(infiles, outfiles):
            logger.info("Processing file %s", infile)
            with file_open(infile, 'r') as inf, file_open(outfile, 'w') as outf:
                for line in tqdm(itertools.islice(inf, num)):
                    outf.write(line)

    def tail(self, parameters, overwrite=False):
        """Take the last n lines from file(s)"""
        self._check_extra_parameters({'inputs', 'outputs', 'n'}, parameters)
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in head")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        num = parameters['n']
        for infile, outfile in zip(infiles, outfiles):
            logger.info("Processing file %s", infile)
            with file_open(infile, 'r') as inf, file_open(outfile, 'w') as outf:
                tmp = []
                for line in tqdm(inf):
                    tmp.append(line)
                    if len(tmp) > num:
                        tmp.pop(0)
                for line in tmp:
                    outf.write(line)

    @classmethod
    def _multipair_gen(cls, lists_of_files):
        """Generator for lines in lists of parallel files"""
        infs = [[file_open(infile) for infile in infiles] for infiles in lists_of_files]
        lines = [[fobj.readline() for fobj in flist] for flist in infs]
        while all(line for linelist in lines for line in linelist):
            yield lines
            lines = [[fobj.readline() for fobj in flist] for flist in infs]
        for folist in infs:
            cls._close_files(*folist)

    def product(self, parameters, overwrite=False):
        """Sample a product of segments from lists of alternative files"""
        self._check_extra_parameters({'inputs', 'outputs', 'k', 'seed', 'skip_empty', 'skip_duplicates'}, parameters)
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

    @staticmethod
    def _lines_to_files(lines, outfiles):
        """Write Nth line to Nth output file"""
        for idx, line in enumerate(lines):
            outfiles[idx].write(line)

    def split(self, parameters, overwrite=False):
        """Split parallel files to two subsets"""
        self._check_extra_parameters(
            {'inputs', 'outputs', 'outputs_2', 'divisor', 'threshold', 'compare', 'hash', 'seed'}, parameters)
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        outfiles_2 = [os.path.join(self.output_dir, fname) for fname in parameters['outputs_2']] \
            if 'outputs_2' in parameters else []
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles) or (outfiles_2 and len(outfiles_2) != len(infiles)):
            raise ConfigurationError("Number of input and output files should match in split")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles + outfiles_2):
            logger.info("Output files exists, skipping step")
            return
        divisor = parameters['divisor']
        threshold = parameters.get('threshold', 1)
        hasher = segment_hash.SegmentHasher(
            compare=parameters.get('compare', 'all'),
            method=parameters.get('hash', 'xx_64'),
            hashseed=parameters.get('seed', 0)
        )
        infs = [file_open(fname) for fname in infiles]
        outfs = [file_open(fname, 'w') for fname in outfiles]
        outfs_2 = [file_open(fname, 'w') for fname in outfiles_2]
        hits = 0
        total = 0
        for lines in tqdm(zip(*infs)):
            total += 1
            if hasher.apply(lines) % divisor < threshold:
                hits += 1
                self._lines_to_files(lines, outfs)
            elif outfs_2:
                self._lines_to_files(lines, outfs_2)
        logger.info("Split %d lines to %s (%.2f%%) and %d (%.2f%%) lines",
                    total, hits, 100 * hits / total, total - hits, 100 * (total - hits) / total)
        self._close_files(*infs)
        self._close_files(*outfs)
        if outfs_2:
            self._close_files(*outfs_2)

    @classmethod
    def _hash_counter(cls, files, hasher):
        """Collect hash values from segment pairs in files"""
        infs = [file_open(infile) for infile in files]
        counter = collections.Counter(hasher.apply(lines) for lines in tqdm(zip(*infs)))
        cls._close_files(*infs)
        return counter

    def remove_duplicates(self, parameters, overwrite=False):
        """Remove duplicates from parallel lines in files"""
        self._check_extra_parameters(
            {'inputs', 'outputs', 'compare', 'hash', 'letters_only', 'lowercase', 'overlap'}, parameters)
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
            method=parameters.get('hash', 'xx_64'),
            letters_only=parameters.get('letters_only', False),
            lowercase=parameters.get('lowercase', False),
        )
        infs = [file_open(infile) for infile in infiles]
        outfs = [file_open(outfile, 'w') for outfile in outfiles]
        overlap = parameters.get('overlap', None)
        if overlap:
            overlap_counter = self._hash_counter(overlap, hasher)
            logger.info("Collected %s types from %s tokens", len(overlap_counter), sum(overlap_counter.values()))
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
            self._lines_to_files(lines, outfs)
        logger.info("Removed %d / %d = %.2f%% duplicate lines (duplicate types: %d)",
                    removed_entries, total, 100 * removed_entries / total if total > 0 else 0,
                    sum(1 for c in counter.values() if c > 1))
        self._close_files(*infs)
        self._close_files(*outfs)

    def unzip(self, parameters, overwrite=False):
        """Unzip parallel segments joined in a single file into multiple files"""
        self._check_extra_parameters({'input', 'outputs', 'separator'}, parameters)
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
                    raise ConfigurationError(f"Number output files do not match the {len(parts)} parts in line {idx}")
                for part, outf in zip(parts, outfs):
                    outf.write(part.strip() + '\n')

    @ParallelWrapper({'inputs', 'outputs', 'preprocessors'})
    def preprocess(self, parameters, overwrite=False):
        """Run preprocessors on text data"""
        # no need to check extra parameters, they are checked in the parallel wrapper
        outfiles = [os.path.join(self.output_dir, fname) for fname in parameters['outputs']]
        infiles = [os.path.join(self.output_dir, fname) for fname in parameters['inputs']]
        if len(outfiles) != len(infiles):
            raise ConfigurationError("Number of input and output files should match in preprocess")
        if not overwrite and all(os.path.isfile(outfile) for outfile in outfiles):
            logger.info("Output files exists, skipping step")
            return
        preprocess_pipe = pipeline.PreprocessorPipeline.from_config(parameters['preprocessors'], workdir=self.output_dir)
        pairs = preprocess_pipe.process(self.pair_generator(*infiles))
        outfileobjs = [file_open(fname, 'w') for fname in outfiles]
        for pair in tqdm(pairs):
            for item, fobj in zip(pair, outfileobjs):
                fobj.write(item + '\n')
                fobj.flush()
        self._close_files(*outfileobjs)

    def download_file(self, parameters, overwrite=False):
        """Download file"""
        self._check_extra_parameters({'output', 'url'}, parameters)
        outfile = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(outfile):
            logger.info("Output file exists, skipping step")
            return
        file_download(parameters['url'], outfile)

    def write_to_file(self, parameters, overwrite=False):
        """Write specified data to file"""
        self._check_extra_parameters({'output', 'data'}, parameters)
        outfile = os.path.join(self.output_dir, parameters['output'])
        if not overwrite and os.path.isfile(outfile):
            logger.info("Output file exists, skipping step")
            return
        data = parameters.get('data', '')
        with file_open(outfile, 'w') as outf:
            outf.write(data)
