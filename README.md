# OpusFilter

OpusFilter is a tool for filtering and combining parallel corpora. It
uses the OpusTool library to download data from the
[OPUS](http://opus.nlpl.eu/) corpus collection, but can be used with
any corpora in raw text format.

## Installing

`pip install .` or `python setup.py install`

### Required libraries

* beautifulsoup4
* langid
* mosestokenizer
* OpusTools
* pycld2
* PyYAML
* regex
* tqdm

### Optional libraries and tools

For using n-gram language model filters, you need to install VariKN
(https://github.com/vsiivola/variKN) and its Python wrapper. The
library files compiled to `build/lib/python` should be added to your
`PYTHONPATH` environment variable.

For using word alignment filters, you need to install elfomal
(https://github.com/robertostling/eflomal) and set environment
variable `EFLOMAL_PATH` to eflomal's root directory, which contains
the Python scripts `align.py` and `makepriors.py`.

## Usage

The package provides a single script, `opus_filter`, that takes a
configuration file as an input. The configuration files are in
[YAML](https://yaml.org/) format. At the top level, they have
to sections:

* `common`, which may include `output_directory` for setting where to
  write the output files. If it is not set, the current working
  directory is used.

* `steps`, which is a list of the corpus processing steps.

The script will run the steps one by one and stops when the final step
has been processed (if no exceptions were raised). The script has
options for setting the last step to run (`--last <N>`) and running
only a single step (`--single <N>`). It the latter, the user has to
make sure that all input files for the step already exist. The first
step has number 1, and -1 points to the last step, -2 to the second to
last, and so on.

By default, existing output files will be re-used, and the steps
producing them skipped. The `--overwrite` option will force overwrite
for all the steps.

Each step in `steps` is a dictionary (mapping) with two keys: `type`
and `parameters`. Type is a string that defines the function that
should be run, and parameters is a dictionary with keys that depend on
the function; at minimum the output files are defined there.

A very simple configuration file that downloads a parallel corpus
(here Finnish-English ParaCrawl v4) from OPUS and stores its segments
to the files `paracrawl.fi.gz` and `paracrawl.en.gz` looks like this:

```
steps:
  - type: opus_read
    parameters:
      corpus_name: ParaCrawl
      source_language: fi
      target_language: en
      release: v4
      preprocessing: raw
      src_output: paracrawl.fi.gz
      tgt_output: paracrawl.en.gz
```

The corpus files processed by OpusFilter are UTF-8 text files that
contain one segment per line. Compressed files are read and written if
the file ends with `.gz` (gzip) or `.bz2` (bzip2).

A bit more complex example that downloads both ParaCrawl and WMT-News
sets from OPUS, concatenates the output files, and filters them so
that only the segment pairs for which both languages have segment
length of 1-100 words and the ratio of the lengths is at most 3
remain:

```
steps:
  - type: opus_read
    parameters:
      corpus_name: ParaCrawl
      source_language: fi
      target_language: en
      release: v4
      preprocessing: raw
      src_output: paracrawl.fi.gz
      tgt_output: paracrawl.en.gz

  - type: opus_read
    parameters:
      corpus_name: WMT-News
      source_language: fi
      target_language: en
      release: v2019
      preprocessing: raw
      src_output: wmt.fi.gz
      tgt_output: wmt.en.gz

  - type: concatenate
    parameters:
      inputs:
      - paracrawl.fi.gz
      - wmt.fi.gz
      output: all.fi.gz

  - type: concatenate
    parameters:
      inputs:
      - paracrawl.en.gz
      - wmt.en.gz
      output: all.en.gz

  - type: filter
    parameters:
      src_input: all.fi.gz
      tgt_input: all.en.gz
      src_output: filtered.fi.gz
      tgt_output: filtered.en.gz
      filters:
        - LengthFilter:
            unit: word
            min_length: 1
            max_length: 100

        - LengthRatioFilter:
            unit: word
            threshold: 3
```


### Available functions

#### Downloading and selecting data

##### `opus_read`

Read a corpus from OPUS collection.

Parameters:

* `corpus_name`: name of the corpus in OPUS
* `source_language`: language code for the source language
* `target_language`: language code for the target language
* `release`: version of the corpus in OPUS
* `preprocessing`: `raw` for untokenized and `xml` for tokenized segments
* `src_output`: output file for source language
* `tgt_output`: output file for target language

##### `concatenate`

Concatenate two text files.

Parameters:

* `inputs`: a list of input files
* `output`: output file

##### `subset`

Take a random subset from parallel corpus files.

Parameters:

* `src_input`: input file for source language 
* `tgt_input`: input file for target language
* `src_output`: output file for source language 
* `tgt_output`: output file for target language
* `seed`: seed for the random generator; set to ensure that two runs select the same lines (default null)
* `size`: number of lines to select to the subset
* `shuffle_target`: take different random lines from the target language; can be used to produce noisy examples for training a corpus filtering model (default false)

#### Filtering and scoring

##### `filter`

Filter parallel data with a combination of filters.

##### `score`

Calculate filtering scores for the lines of parallel data.

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

Train word alignment priors for eflomal. Can be used in `WordAlignFilter`.

### Available filters

#### Length filters

##### `LengthFilter`

##### `LengthRatioFilter`

#### Script and language identification filters

##### `CharacterScoreFilter`

##### `LanguageIDFilter`

#### Special character filters

##### `HtmlTagFilter`

##### `TerminalPunctuationFilter`

##### `NonZeroNumeralsFilter`

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

##### `WordAlignFilter`
