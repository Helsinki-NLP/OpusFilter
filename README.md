# OpusFilter

OpusFilter is a tool for filtering and combining parallel corpora. It
uses the OpusTool library to download data from the
[OPUS](http://opus.nlpl.eu/) corpus collection, but can be used with
any corpora in raw text format.

## Table of contents

* [Installing](#installing)
   * [Required libraries](#required-libraries)
   * [Optional libraries and tools](#optional-libraries-and-tools)
* [Usage](#usage)
   * [Available functions](#available-functions)
      * [Downloading and selecting data](#downloading-and-selecting-data)
         * [opus_read](#opus_read)
         * [concatenate](#concatenate)
         * [subset](#subset)
      * [Filtering and scoring](#filtering-and-scoring)
         * [filter](#filter)
         * [score](#score)
      * [Training models](#training-models)
         * [train_ngram](#train_ngram)
         * [train_aligment](#train_aligment)
   * [Available filters](#available-filters)
      * [Length filters](#length-filters)
         * [LengthFilter](#lengthfilter)
         * [LengthRatioFilter](#lengthratiofilter)
      * [Script and language identification filters](#script-and-language-identification-filters)
         * [CharacterScoreFilter](#characterscorefilter)
         * [LanguageIDFilter](#languageidfilter)
      * [Special character filters](#special-character-filters)
         * [HtmlTagFilter](#htmltagfilter)
         * [TerminalPunctuationFilter](#terminalpunctuationfilter)
         * [NonZeroNumeralsFilter](#nonzeronumeralsfilter)
      * [Language model filters](#language-model-filters)
         * [CrossEntropyFilter](#crossentropyfilter)
      * [Alignment model filters](#alignment-model-filters)
         * [WordAlignFilter](#wordalignfilter)
   * [Custom filters](#custom-filters)

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
(https://github.com/vsiivola/variKN) and its Python wrapper. Include
the library files compiled to `build/lib/python` to your Python
library path (e.g. by setting the `PYTHONPATH` environment variable).

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

```yaml
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

```yaml
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

YAML node anchors (`&name`) and references (`*name`) can be used. They
are especially useful when defining a set of filters you want to use
for different data sets. For example, if in the previous example you
wanted to use the same filters separately for the ParaCrawl and
WMT-News data, you can have:

```yaml
  - type: filter
    parameters:
      src_input: paracrawl.fi.gz
      tgt_input: paracrawl.en.gz
      src_output: paracrawl_filtered.fi.gz
      tgt_output: paracrawl_filtered.en.gz
      filters: &myfilters
        - LengthFilter:
            unit: word
            min_length: 1
            max_length: 100

        - LengthRatioFilter:
            unit: word
            threshold: 3

  - type: filter
    parameters:
      src_input: wmt.fi.gz
      tgt_input: wmt.en.gz
      src_output: wmt_filtered.fi.gz
      tgt_output: wmt_filtered.en.gz
      filters: *myfilters
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

Parameters:

* `src_input`: input file for source language
* `tgt_input`: input file for target language
* `src_output`: output file for source language
* `tgt_output`: output file for target language
* `filters`: a list of filters to apply; see below
* `filterfalse`: Instead of keeping segment pairs that match all the filters, keep those that match none of the filters (default false)

The filters parameter is a list of dictionaries, each representing one
filter. The top level should typically include a single key that
defines the class name for the filter (e.g. `LenghtFilter`).
Additionally it can include a special key `module` for defining module
name for custom filters (see the end of the document for details).

Under the class name there is a dictionary the defines the parameters
of the filters. The are mostly specific to the filter class; see the
section Available filters for ready-made filters. An exception is a
parameter `name` that is available for all filters. It has no effect
for the filter function, but is useful for the score function below.

The output of the step is only those segment pairs that are accepted
by all the filters (unless `filterfalse` is set true, in which case
the output is those segment pairs that are rejected by all the
filters; note that this is not the opposite result of the default
behaviour).

##### `score`

Calculate filtering scores for the lines of parallel data.

Parameters:

* `src_input`: input file for source language
* `tgt_input`: input file for target language
* `output`: output file for the scores
* `filters`: a list of filters to apply; see below

The filters are defined in the same manner as in the `filter`
function. The possible accept threshold parameters of the filters do
not have an effect in scoring; each filter simply outputs one or more
numerical values.

The scores are written in the JSON Lines format: each line contains a
single JSON object. The top level of the object contains class names
for the filters. If there is only of filter of the specific class, and
its score is a single number, the value of the score is simply below
the class name. The the filter outputs more scores, they are
represented in a dictionary. Typically there is a key `src` for source
segment score and `tgt` for target segment score, but the number of
scores and their keys are not restricted.

The filters may contain the same filter class multiple times so that
the same filter can be used with different parameters (e.g. both words
and characters as units for length-based filters). In this case, under
the top-level filter class key there is another dictionary that
separates the filter instances. The keys for the instances can be
defined by using the `name` parameter that is available for all
filters. If the name is not defined, the first filter of the class is
given key "1", the second "2", and so on. (Note: Make sure to give a
name to either all or none of the filters, or at least do not manually
give integers as names.)

The output can be used e.g. for analyzing the distribution of the
scores or training a classifier for filtering. The JSON Lines data
is easy to load as a [pandas](https://pandas.pydata.org/) DataFrame using the [`json_normalize`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.json.json_normalize.html)
method.

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

Filtering based on absolute segment lengths.

Parameters:

* `min_length`: Minimum segment length (default 1)
* `max_length`: Maximum segment length (default 100)
* `unit`: Type of unit for calculating the lengths: `word` for words (or any whitespace-separated units) and `character` or `char` for characters. The default is `word`.

Returned scores are lengths for the source and target segment. In
filtering, both segments have to be between the minimum and maximum
length thresholds.

##### `LengthRatioFilter`

Filtering based on ratio of the segment lengths.

Parameters:

* `threshold`: Threshold for the length ratio
* `unit`: Type of unit for calculating the lengths: `word` for words (or any whitespace-separated units) and `character` or `char` for characters. The default is `word`.

Returned score is the higher length divided by the lower length, or
infinity of either of the lengths are zero. In filtering, segment
pairs is accepted of the ratio is below the given threshold.

#### Script and language identification filters

##### `CharacterScoreFilter`

##### `LanguageIDFilter`

#### Special character filters

##### `HtmlTagFilter`

##### `TerminalPunctuationFilter`

##### `NonZeroNumeralsFilter`

#### Language model filters

##### `CrossEntropyFilter`

Filter segments by n-gram language model probabilities.

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

Note that the format and boundary marking should match the parameters
used in model training.

Separate scores (entropy, perplexity, or negative log-probability) are
returned for the source and target segment. In filtering, the segment
pair is accepted if both values are below the respective thresholds,
and their absolute difference is below the difference threshold.

#### Alignment model filters

##### `WordAlignFilter`


### Custom filters

You can also import your own filters by defining the `module` key in
the filter configuration entries.

The custom filters should inherit the abstract base class `FilterABC`
from the `opusfilter` package. They should implement two abstract
methods: `score` and `accept`.

The `score` method is a generator that takes an iterator over tuples
of parallel sentences, and yields a score object for each pair. The
score may either be a single number, or if multiple score values need
to be yielded, a dictionary that has the numbers as values.

The `accept` method takes a single output yielded by the `score`
method, and returns whether the sentence pair should be accepted based
on the score.

If the filter requires any parameters (e.g. score thresholds for the
`accept` method), the class should implement also the `__init__`
method.  Arbitrary keyword arguments should be accepted (with
`**kwargs`), and the `__init__` method of the base class (`FilterABC`)
should be called in the end with the remaining keyword arguments. The
keyword argument `name` is reserved for giving names to the filters.

The base class defines the methods `decisions`, `filter`, and
`filterfalse` based on the `score` and `accept` methods; they should
not be redefined except for a good reason.

The example below shows code for simple filter that calculates the
proportion of uppercase letters in the sentences, and accepts the pair
only if both sentences have less than 50% (or given threshold) of
uppercase characters:

```python
import opusfilter

class UppercaseFilter(opusfilter.FilterABC):

    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def score(self, pairs):
        for sent1, sent2 in pairs:
            length1 = len(sent1)
            length2 = len(sent2)
            up1 = sum(1 for c in sent1 if c.isupper()) / length1 if length1 > 0 else 0
            up2 = sum(1 for c in sent2 if c.isupper()) / length2 if length2 > 0 else 0
            yield {'src': up1, 'tgt': up2}

    def accept(self, score):
        up1, up2 = score['src'], score['tgt']
        return up1 < self.threshold and up2 < self.threshold
```

Assuming that the above code is in a module named `customfilter` in
the Python evironment (e.g. save the code as `customfilter.py` and add
the directory that contains it to `PYTHONPATH` environment variable),
it can be selected in the filter configurations as follows:

```yaml
steps:

  ...

  - type: filter
    parameters:

      ...

      filters:

        - UppercaseFilter:
            threshold: 0.5
          module: customfilter
```
