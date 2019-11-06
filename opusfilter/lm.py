"""Language model filtering"""

import argparse
import copy
import logging
import math
import tempfile

from . import FilterABC, ConfigurationError


logger = logging.getLogger(__name__)


try:
    import varikn
except ImportError:
    logger.warning("Could not load varikn, language model filtering not supported")


_VARIKN_TRAINING_PARAMS = {
    'optdata': '',
    'norder': 0,
    'dscale': 0.001,
    'dscale2': 0,
    'arpa': True,
    'use_3nzer': False,
    'absolute': False,
    'cutoffs': '0 0 1'
}


def train(datafile, outputfile, **kwargs):
    """Train varigram language model with VariKN

    Positional arguments:
      datafile -- Filename for training corpus
      outpufile -- Filename for trained model

    Keyword arguments:
      optdata -- Filename for optimization data
      norder -- Limit model order (0 = no limit)
      dscale -- Model size scale factor
      dscale2 -- Model size scaling during pruning step (default no pruning=0)
      arpa -- Output arpa instead of binary LM
      use_3nzer -- Use 3 discounts per order instead of one
      absolute -- Use absolute discounting instead of Kneser-Ney smoothing
      cutoffs -- Use the specified cutoffs (default "0 0 1"). The last value is used
        for all higher order n-grams.

    Any extra keyword arguments are ignored.

    """
    args = argparse.Namespace()
    for key, default in _VARIKN_TRAINING_PARAMS.items():
        setattr(args, key, kwargs.get(key, default))
    trainer = varikn.VarigramTrainer(args.use_3nzer, args.absolute)
    trainer.set_datacost_scale(args.dscale)
    trainer.set_datacost_scale2(args.dscale2)
    if args.norder > 0:
        trainer.set_max_order(args.norder)
    # initialize(infilename, hashs, ndrop, nfirst, optiname, "<s>", smallmem, vocabname)
    trainer.initialize(datafile, 0, 0, -1, args.optdata, '<s>', False, '')
    trainer.set_cutoffs([int(x) for x in args.cutoffs.split()])
    trainer.grow(1)
    trainer.write_file(outputfile, args.arpa)


def token_perplexity(lm, tokens):
    """Calculate token perplexity, entropy, and negative logprob for sentence tokens"""
    lpsum = 0.0
    for token in tokens:
        lpsum += lm.token_logprob(token)
    logprob = -lpsum / math.log10(2)
    entropy = logprob / lm.processed_tokens()
    ppl = 10**(-lpsum / lm.processed_tokens())
    lm.clear_history()
    lm.init_variables()
    return ppl, entropy, logprob


def word_perplexity(lm, tokens):
    """Calculate word perplexity, entropy, and negative logprob for sentence tokens"""
    lpsum = 0.0
    for token in tokens:
        lpsum += lm.word_logprob(token)
    logprob = -lpsum / math.log10(2)
    entropy = logprob / lm.processed_words()
    ppl = 10**(-lpsum / lm.processed_words())
    lm.clear_history()
    lm.init_variables()
    return ppl, entropy, logprob


_VARIKN_PERPLEXITY_PARAMS = {
    'arpa': True,
    'unk': '<UNK>',
    'include_unks': False,
    'ccs': None,
    'wb': '<w>',
    'mb': None,
    'init_hist': 2,
    'interpolate': None,
    'filename': None
}


def get_perplexity_params(params):
    new = copy.copy(_VARIKN_PERPLEXITY_PARAMS)
    new.update(params)
    return new


def _temptokenfile(item):
    _, tmpfname = tempfile.mkstemp()
    with open(tmpfname, 'w') as fobj:
        fobj.write("{}\n".format(item))
    return tmpfname


def get_lm(**kwargs):
    """Return language model initialized for perplexity calculation

    Keyword arguments:
      filename -- Filename for the language model to use
      arpa -- LM is in arpa format instead of binary LM (default: True)
      unk -- Unk symbol (default: '<UNK>', case sensitive)
      include_unks -- Include unknown tokens in perplexity calculations (default: False)
      ccs -- List of context cues ignored in perplexity calculations (default: None)
      mb -- Morph boundary marking (default '')
      wb -- Word boundary tag (default '<w>')
      init_hist -- Ignore n first tokens after "</s>" in perplexity calculations (default: 2)
      interpolate -- List of language models (arpa format) and interpolation weights (default: None)

    Any extra keyword arguments are ignored.

    """

    args = argparse.Namespace()
    for key, default in _VARIKN_PERPLEXITY_PARAMS.items():
        setattr(args, key, kwargs.get(key, default))

    args.ccs = _temptokenfile(args.ccs) if args.ccs else ''
    args.wb = _temptokenfile(args.wb) if args.wb else ''
    args.mb = _temptokenfile(args.mb) if args.mb else ''

    if args.interpolate:
        lms = [args.filename] + [x[0] for x in args.interpolate]
        wsum = sum(float(x[1]) for x in args.interpolate)
        if wsum >= 1:
            logger.warning("Weights are too high!")
        weights = [1 - wsum] + [float(x[1]) for x in args.interpolate]
        tgram = varikn.InterTreeGram(lms, weights)
        lm = varikn.Perplexity(
            tgram, args.ccs, args.wb, args.mb, args.unk, not args.include_unks)
    else:
        lm = varikn.Perplexity(
            args.filename, 0 if args.arpa else 1,
            args.ccs, args.wb, args.mb, args.unk, 0, not args.include_unks)
    lm.set_init_hist(args.init_hist)
    lm.init_variables()
    return lm


class LMTokenizer:
    """Tokenizer for subword language models"""

    s_beg = '<s>'
    s_end = '</s>'

    def __init__(self, segmentation=None, mb='', wb='<w>', **kwargs):
        """Map sentences to tokens processed by language models

        Keyword arguments:
          segmentation -- Word segmentation parameters; currently only {'type': 'char'} is
            supported (default: None)
          mb -- Morph boundary marking (default '')
          wb -- Word boundary tag (default '<w>')

        Any extra keyword arguments are ignored.

        """
        if segmentation and segmentation.get('type', 'char') != 'char':
            raise ConfigurationError("Only segmentation type supported currently is 'char'")
        self.mb = mb
        self.wb = wb

    def tokenize(self, sent):
        """Tokenize single sentence"""
        tokens = [self.s_beg]
        if self.wb:
            tokens.append(self.wb)
        for word in sent.strip().split():
            if self.mb and self.mb.endswith('$'):
                for char in word.replace('', self.mb[:-1] + ' '):
                    tokens.append(char)
            elif self.mb and self.mb.startswith('^'):
                for char in word.replace('', ' ' + self.mb[1:]):
                    tokens.append(char)
            else:
                tokens += list(word)
            if self.wb:
                tokens.append(self.wb)
        tokens.append(self.s_end)
        return tokens


class CrossEntropyFilter(FilterABC):
    """Filtering based on language model scores"""

    score_types = {'entropy', 'perplexity', 'logprob'}

    def __init__(self, src_lm_params=None, tgt_lm_params=None, score_type='entropy',
                 src_threshold=50.0, tgt_threshold=50.0, diff_threshold=10.0, **kwargs):
        if not src_lm_params or not tgt_lm_params:
            raise ConfigurationError("Language model configurations need to be defined")
        if src_lm_params.get('segmentation', {}).get('type', 'char') != 'char':
            raise ConfigurationError("Only segmentation type supported currently is 'char'")
        if score_type not in self.score_types:
            raise ConfigurationError("Unknown score type {}, should be one of {}".format(
                score_type, self.score_types))
        self.score_type = score_type
        self.src_lm_params = src_lm_params
        self.tgt_lm_params = tgt_lm_params
        self.src_lm = get_lm(**self.src_lm_params)
        self.tgt_lm = get_lm(**self.tgt_lm_params)
        self.src_threshold = src_threshold
        self.tgt_threshold = tgt_threshold
        self.diff_threshold = diff_threshold
        super().__init__(**kwargs)

    def accept(self, score):
        src, tgt = score['src'], score['tgt']
        diff = abs(src - tgt)
        return src < self.src_threshold and tgt < self.tgt_threshold and diff < self.diff_threshold

    def score(self, pairs):
        src_tokenizer = LMTokenizer(**self.src_lm_params)
        tgt_tokenizer = LMTokenizer(**self.tgt_lm_params)
        for sent1, sent2 in pairs:
            scores = {}
            for key, lm, tokenizer, sent in [('src', self.src_lm, src_tokenizer, sent1),
                                             ('tgt', self.tgt_lm, tgt_tokenizer, sent2)]:
                tokens = tokenizer.tokenize(sent)
                use_word = tokenizer.wb or tokenizer.mb
                ppl, entr, logprob = word_perplexity(lm, tokens) if use_word \
                                     else token_perplexity(lm, tokens)
                if self.score_type == 'logprob':
                    scores[key] = logprob
                elif self.score_type == 'perplexity':
                    scores[key] = ppl
                else:
                    scores[key] = entr
            yield scores
