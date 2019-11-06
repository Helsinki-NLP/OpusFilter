# OpusFilter

Tool for filtering parallel corpora.

## Installing

`pip install`

### Optional requirements

For using n-gram language model filters, you need to install VariKN
(https://github.com/vsiivola/variKN) and its Python bindings.

For using word alignment filters, you need to install elfomal
(https://github.com/robertostling/eflomal) and set environment
variable `EFLOMAL_PATH` to eflomal's root directory, which contains
the Python scripts `align.py` and `makepriors.py`.

## Filter configuration files

### Available functions

#### Downloading and selecting data

##### `opus_read`

##### `concatenate`

##### `subset`

#### Filtering and scoring

##### `filter`

##### `score`

#### Training models

##### `train_ngram`

Train a character-based varigram language model with VariKN. Can be used for `CrossEntropyFilter`.

Parameters:

* `data`: Input file name for training data
* `model`: Output file name for the model
* `parameters`: Training options for VariKN and tokenization
   * `optdata`: Filename for optimization data (default empty = use leave-one-out estimation instead)
   * `norder`: Limit model order (default 0 = no limit)
   * `dscale`: Model size scale factor (smaller value gives a larger model; default 0.001)
   * `dscale2`: Model size scaling during pruning step (default 0 = no pruning)
   * `arpa`: Output ARPA instead of binary LM (default true)
   * `use_3nzer`: Use 3 discounts per order instead of one (default false)
   * `absolute`: Use absolute discounting instead of Kneser-Ney smoothing (default false)
   * `cutoffs`: Use the specified cutoffs (default "0 0 1"). The last value is used for all higher order n-grams.
   * `mb`: Word-internal boundary marking (default `''`)
   * `wb`: Word boundary tag (default `'<w>'`)

See [VariKN](https://github.com/vsiivola/variKN) documentation for details.

##### `train_aligment`


### Available filters

#### Length filters

#### Script and language identification filters

#### Special character filters

#### Language model filters

##### `CrossEntropyFilter`

Parameters:

* `src_lm_params`: dictionary for the parameters for the source language model; see below
* `tgt_lm_params`: dictionary for the parameters for the target language model; see below
* `score_type`: select whether to calculate cross-entropy (`entropy`; default), perplixty (`perplexity`) or negative log-probability (`logprob`) scores
* `src_threshold`: upper threshold for source language score when filtering (default 50.0)
* `tgt_threshold`: upper threshold for target language score when filtering (default 50.0)
* `diff_threshold`: upper threshold for absolute difference of source and target language scores when filtering (default 10.0)

Language model paramters for `src_lm_params` and `tgt_lm_params`:

* `filename`: Filename for the language model to use
* `arpa`: LM is in ARPA format instead of binary LM (default: true)
* `unk`: Unk symbol (default: `<UNK>`, case sensitive)
* `include_unks`: Include unknown tokens in perplexity calculations (default: false)
* `ccs`: List of context cues ignored in perplexity calculations (default: none)
* `mb`: Morph boundary marking (default `''`)
* `wb`: Word boundary tag (default `'<w>'`)
* `init_hist`: Ignore n first tokens after `</s>` in perplexity calculations (default: 2)
* `interpolate`: List of language models (in ARPA format) and interpolation weights (default: none)

Note that the format and boundary marking should match the parameters used in model training.

#### Alignment model filters
