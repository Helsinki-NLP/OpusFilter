# OpusFilter

OpusFilter is a tool for filtering and combining parallel corpora. It
uses the OpusTool library to download data from the OPUS corpus
collection, but can be used with any corpora in raw text format.

Features:

* Preprocessing pipelines configured with [YAML](https://yaml.org/)
* Simple downloading of parallel corpora from [OPUS](http://opus.nlpl.eu/)
* Implementations for many common text file operations on parallel files
* Memory-efficient processing of large files
* Implemented filters based e.g. on language identification, word
  aligment, and n-gram language models
* Extendable with your own filters written in Python

OpusFilter has been presented in [ACL 2020 system demonstrations](https://www.aclweb.org/anthology/2020.acl-demos.20).

## Table of contents

* [Installing](#installing)
   * [Required libraries](#required-libraries)
   * [Optional libraries and tools](#optional-libraries-and-tools)
* [Citing](#citing)
* [Overview](#overview)
   * [Examples](#examples)
* [Available functions](#available-functions)
   * [Downloading and selecting data](#downloading-and-selecting-data)
      * [opus_read](#opus_read)
      * [concatenate](#concatenate)
      * [head](#head)
      * [tail](#tail)
      * [slice](#slice)
      * [split](#split)
      * [subset](#subset)
   * [Filtering and scoring](#filtering-and-scoring)
      * [remove_duplicates](#remove_duplicates)
      * [filter](#filter)
      * [score](#score)
   * [Using score files](#using-score-files)
      * [join](#join)
      * [sort](#sort)
   * [Training language and alignment models](#training-language-and-alignment-models)
      * [train_ngram](#train_ngram)
      * [train_aligment](#train_aligment)
   * [Training and using classifiers](#training-and-using-classifiers)
      * [train_classifier](#train_classifier)
      * [classify](#classify)
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
* [Other tools](#other-tools)

## Installing

`pip install .` or `python setup.py install`

### Required libraries

* beautifulsoup4
* langid
* mosestokenizer
* OpusTools
* pandas
* pycld2
* pyhash
* PyYAML
* regex
* scikit-learn
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

## Citing

If you use OpusFilter in your research, please cite our [ACL 2020 paper](https://www.aclweb.org/anthology/2020.acl-demos.20):

```
@inproceedings{aulamo-etal-2020-opusfilter,
    title = "{O}pus{F}ilter: A Configurable Parallel Corpus Filtering Toolbox",
    author = {Aulamo, Mikko and Virpioja, Sami and Tiedemann, J{\"o}rg},
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-demos.20",
    doi = "10.18653/v1/2020.acl-demos.20",
    pages = "150--156"
}
```

## Overview

The package provides the script `opusfilter` that takes a
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

### Examples

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

## Available functions

### Downloading and selecting data

#### `opus_read`

Read a corpus from OPUS collection.

Parameters:

* `corpus_name`: name of the corpus in OPUS
* `source_language`: language code for the source language
* `target_language`: language code for the target language
* `release`: version of the corpus in OPUS
* `preprocessing`: `raw` for untokenized and `xml` for tokenized segments
* `src_output`: output file for source language
* `tgt_output`: output file for target language

#### `concatenate`

Concatenate two or more text files.

Parameters:

* `inputs`: a list of input files
* `output`: output file

#### `head`

Take the first n lines from files.

Parameters:

* `inputs`: a list of input files
* `outputs`: a list of output files
* `n`: number of output lines

#### `tail`

Take the last n lines from files.

Parameters:

* `inputs`: a list of input files
* `outputs`: a list of output files
* `n`: number of output lines

Note: The memory requirement of `tail` is proportional to n. Use
`slice` if you need all except the first n lines.

#### `slice`

Take slice of lines from files.

Parameters:

* `inputs`: a list of input files
* `outputs`: a list of output files
* `start`: start index (optional; default 0)
* `stop`: stop index (optional; default `null`)
* `step`: step size (optional; default 1)

Either `start`, `stop`, or both of them should be given. If `stop` is
not given, reads until the end of the file.

#### `split`

Split files to two parts giving the approximative proportions as
fractions.

Parameters:

* `inputs`: input file(s)
* `outputs`: output file(s) for selected lines
* `outputs_2`: output file(s) for the rest of the lines (optional)
* `divisor`: divisor for the modulo operation (e.g. 2 for splitting to equal sized parts)
* `threshold`: threshold for the output of the modulo operation (optional; default 1)
* `compare`: select files to use for hash operation (optional; default `all` or a list of indices)
* `hash`: select hash algorithm from pyhash (optional; default `xx_64`)
* `seed`: integer seed for the hash algorithm (optional; default 0)

Input files are processed line by line in parallel. If the condition
`hash(content) % divisor < threshold`, where the content is a
concatenation of the input lines and the hash function returns an
integer, holds, the lines are written to the `outputs`. If the
condition does not hold, and `outputs_2` are defined, the lines are
written there.

Compared to random splitting (see [subset](#subset)) or using the
modulo operation on the line number, the benefit of the hash-based
approach is that the decision is fully deterministic and based only on
the *content* of the lines. Consequently, identical content always
goes to the the same output file(s). For example, if you split a
parallel corpus into test and training sets, and you can be sure that
your test data does not contain exactly same samples as the training
data even if the original data has duplicates.

The downside is that you need to be careful if you use several splits
for the same data. The divisors used in consecutive splits should not
themselves have common divisors, or the proportion of the data in the
output files may be unexpected. Distinct prime numbers are good
choices. Also setting a different `seed` value for the hash functions
prevents the issue.

The `compare` parameter can be used to select which input files are
used to generate the content for the hash function. For example, if
you have source and target language files, and you want that the split
depends only on the source or target sentence, set `compare` to `[0]`
or `[1]`, respectively.

#### `subset`

Take a random subset from parallel corpus files.

Parameters:

* `src_input`: input file for source language
* `tgt_input`: input file for target language
* `src_output`: output file for source language
* `tgt_output`: output file for target language
* `size`: number of lines to select for the subset
* `seed`: seed for the random generator; set to ensure that two runs select the same lines (optional; default `null`)
* `shuffle_target`: take different random lines from the target language; can be used to produce noisy examples for training a corpus filtering model (optional; default `false`)

### Filtering and scoring

#### `remove_duplicates`

Filter out duplicate lines from parallel corpus files.

Parameters:

* `inputs`: input file(s)
* `outputs`: output file(s)
* `compare`: select files for duplicate comparison (optional; default `all` or a list of indices)
* `hash`: select hash algorithm from pyhash (optional; default `xx_64`)

Duplicate filtering is recommended as a first step especially if you
combine different corpus collections (e.g. data crawled from web) and
cannot be sure that the same data sources have not been used in many
of them.

The `remove_duplicates` function works for any number of files. The
`compare` parameter can be used to select which input files are used
to generate the key for duplicate comparison. For example, if you have
source and target language files, and you want that each source or
target sentence occurs only once, set `compare` to `[0]` or `[1]`,
respectively.

Non-cryptographic hashing is used to reduce memory consumption for the
case that the files are very large. The lines defined by the `compare`
option are concatenated together, and the hash algorithm is applied on
the result to produce the final key for storing the counts. You can
use any of the hash algorithms implemented in the pyhash library. The
default 64-bit XXHash algorithm should be fine for any practical data
sizes, but if do not care about memory use and want to be extra sure
there are no collisions, you can disable hashing by setting the `hash`
parameter as empty string or null.

#### `filter`

Filter parallel data with a combination of filters.

Parameters:

* `src_input`: input file for source language
* `tgt_input`: input file for target language
* `src_output`: output file for source language
* `tgt_output`: output file for target language
* `filters`: a list of filters to apply; see below
* `filterfalse`: Yield segment pairs that do not pass at least one of the filters (optional; default `false`)

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
by all the filters, unless `filterfalse` is set true, in which case
the output is the opposite (i.e., those segment pairs that are
rejected by at least one filter).

#### `score`

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
given key `"1"`, the second `"2"`, and so on. (Note: make sure to give a
name to either all or none of the filters, or at least do not manually
give integers as names.)

The output can be used e.g. for analyzing the distribution of the
scores or training a classifier for filtering. The JSON Lines data
is easy to load as a [pandas](https://pandas.pydata.org/) DataFrame using the [`json_normalize`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.json.json_normalize.html)
method.

### Using score files

#### `join`

Join two or more score files.

Parameters:

* `inputs`: input files containing scores in JSON Lines format
* `output`: output file for joined scores
* `keys`: a list containing dictionary keys for each input file (optional; default `null`)

If the list of keys is provided, the input objects are inserted under
the corresponding key. The objects can also be inserted deeper in a
hierarchical score dictionary by using a key that has dot-separated
parts. For example, `x.y` means setting key `y` under the key `x`. If
the keys are not provided, or the key corresponding to the input file
is `null`, output object will be updated with the input object and
existing keys will be overwritten.

For example, if you have scores for the source and target sentences
created by external tools (`myscores.src` and `myscores.tgt`
containing one number per line), and you want to join them with an
existing score file created by OpusFilter (`scores.jsonl.gz`), you can
do it like this:

```
  - type: join
    parameters:
      inputs:
      - scores.jsonl.gz
      - myscores.src
      - myscores.tgt
      keys:
      - null
      - MyScore.src
      - MyScore.tgt
      output: scores-joined.jsonl.gz
```

Apart from the old scores from `scores.jsonl.gz`, each line should now
contain `{"MyScore": {"src": ..., "tgt": ...}}`.

#### `sort`

Sort files based on score values.

Parameters:

* `inputs`: input files to sort
* `outputs`: sorted output files
* `values`: input file for values used in sorting
* `reverse`: `true` for descending sort (optional; default `false`)
* `key`: if values file contain JSON objects, use the key to select field (optional; default `null`)
* `type`: force type conversion for the value (optional; `float`, `int`, `str`, or default `null`)

The values file should contain one JSON object per line. If a line
cannot be interpreted as a JSON object, it is read as a plain unicode
string. Dots (`.`) in the key are interpreted as multiple get operations
(e.g. `x.y` expects that there is key `x` under the key `y`). The type
conversion can be used e.g. for forcing numerical values to be compared
as strings.

### Training language and alignment models

#### `train_ngram`

Train a character-based varigram language model with VariKN. Can be used for `CrossEntropyFilter`.

Parameters:

* `data`: input file name for training data
* `model`: output file name for the model
* `parameters`: training options for VariKN and tokenization
   * `optdata`: filename for optimization data (optional; default empty string `""` = use leave-one-out estimation instead)
   * `norder`: limit model order (optional; default 0 = no limit)
   * `dscale`: model size scale factor (optional; smaller value gives a larger model; default 0.001)
   * `dscale2`: model size scaling during pruning step (optional; default 0 = no pruning)
   * `arpa`: output ARPA instead of binary LM (optional; default `true`)
   * `use_3nzer`: use 3 discounts per order instead of one (optional; default `false`)
   * `absolute`: use absolute discounting instead of Kneser-Ney smoothing (optional; default `false`)
   * `cutoffs`: use the specified cutoffs (optional; default `"0 0 1"`). The last value is used for all higher order n-grams.
   * `mb`: word-internal boundary marking (optional; default `""`)
   * `wb`: word boundary tag (optional; default `"<w>"`)

Apart from the scale, cutoff, and order parameters the size of the
model depends on the size of the training data. Typically you want to
at least change the `dscale` value to get a model of a reasonable
size. If unsure, start with high values, look at the number of the
n-grams in the output file, and divide by 10 if it looks too small.
The `dscale2` option is useful mostly if you want to optimize the
balance between the model size and accuracy at the cost of longer
training time; a suitable rule of thumb is double the value of
`dscale`.

The default boundary settings are suitable for character-based models
and are not recommended to edit.

See [VariKN](https://github.com/vsiivola/variKN) documentation for
details.

#### `train_aligment`

Train word alignment priors for eflomal. Can be used in `WordAlignFilter`.

Parameters:

* `src_data`: input file for the source language
* `tgt_data`: input file for the target language
* `parameters`: training options for the aligment and tokenization
   * `src_tokenizer`: tokenizer for source language (optional; default `null`)
   * `tgt_tokenizer`: tokenizer for target language (optional; default `null`)
   * `model`: eflomal model type (optional; default 3)
* `output`: output file name for the priors

See [WordAlignFilter](#wordalignfilter) for details of the training
parameters.

### Training and using classifiers

#### `train_classifier`

Train an `sklearn` classifier to produce a cleanness score for sentence pairs.

Parameters:

* `training_scores`: a file containing filter scores for training in JSON lines format produced with `score` function.
* `criterion`: criterion to be used in classifier optimization (valid options are `CE`, `ROC_AUC`, `SSE`, `AIC` and `BIC`)
* `dev_scores`: a file containing filter scores for training in JSON lines format produced with `score` function with and added item `label` added to each entry. `label` has value 1 for clean pairs and 0 for noisy pairs (optional; `dev_scores` is only used when the `criterion` is `ROC_AUC`)
* `model_type`: classifier model type selected from `sklearn` classifiers (default `LogisticRegression`)
* `model_parameters`: parameters for the `sklearn` classifier
* `model`: output model file
* `features`: the features given to the classifier to be trained on, defined as a list of filter names
    * `ExampleFilter`:
        * `clean-direction`: the direction that indicates higher cleanness (valid options are `high` and `low`)
        * `quantiles`: a dictionary the items of which (`min`, `max` and `initial`) specify the minimum, maximum and inital quantile value that are used in classifier optimization to select negative and positive training examples (default `{'min': 0, 'max': 1, 'initial': 0.1}`)

The classifier is optimized by training multiple classifier model with the training data divided differently into positive and negative examples based on the quantile boundaries specified in each feature. The model that achieves the highest criterion score is then saved in the output file.

#### `classify`

Use a classifier model trained with `train_classifier` to assign a cleanness score or label to sentence pairs that have been scored with `score`.

Parameters:

* `model`: classifier model trained with `train_classifier`
* `scores`: scores of the sentence pairs to be classifed in JSON lines format produced with the `score` function
* `output_probabilities`: file to write the cleanness scores to, 1 is cleanest and 0 is noisiest (optional)
* `output_labels`: file to write the cleanness labels to, 1 is a clean and 0 is a noisy pair (optional)

The probabilities and labels are written to the output files line by line, corresponding to the scores on each line in `scores`.

## Available filters

### Length filters

#### `LengthFilter`

Filtering based on absolute segment lengths.

Parameters:

* `min_length`: minimum segment length (optional; default 1)
* `max_length`: maximum segment length (optional; default 100)
* `unit`: type of unit for calculating the lengths (optional; `word` for words or any whitespace-separated units, and `character` or `char` for characters; the default is `word`)

Returned scores are lengths for the source and target segment. In
filtering, both segments have to be between the minimum and maximum
length thresholds.

#### `LengthRatioFilter`

Filtering based on ratio of the segment lengths.

Parameters:

* `threshold`: threshold for the length ratio
* `unit`: type of unit for calculating the lengths (optional; `word` for words or any whitespace-separated units, and `character` or `char` for characters; the default is `word`)

Returned score is the higher length divided by the lower length, or
infinity of either of the lengths are zero. In filtering, segment
pairs is accepted of the ratio is below the given threshold.

### Script and language identification filters

#### `CharacterScoreFilter`

Filter segments based on what proportion of their alphabetic characters are in a given script. For a list of valid scripts, see e.g. https://www.regular-expressions.info/unicode.html

Parameters:

* `src_script`: script for source segment (default Latin)
* `tgt_script`: script for target segment (default Latin)
* `src_threshold`: minimum proportion of source characters in a script (default 1)
* `tgt_threshold`: minimum proportion of target characters in a script (default 1)

Returned scores are proportions of valid characters in source and target segments. In filtering, both values have to be equal to or greater than the minimum thresholds.

#### `LanguageIDFilter`

Filter segments based on their language identification confidence scores.

Parameters:

* `src_lang`: expected language for source segment
* `tgt_lang`: expected language for target segment
* `id_method`: language indentification method (`langid` for using the `langid` library of `cld2` for using the `cld2` library; the default is `langid`)
* `src_threshold`: minimum identification confidence score for source segment
* `tgt_threshold`: minimum identification confidence score for target segment

Returned scores are the language identification confidence scores from a given identification method for source and target segments. The scores range from 0 to 1. In filtering, both values have to be greater than the minimum thresholds.

### Special character filters

#### `HtmlTagFilter`

Filter segments based on whether they contain HTML tags or not.

The returned scores are two boolean values indicating whether the source and target segments contain HTML tags. In filtering, a segment pair is accepted if neither of the segments contains HTML tags.

#### `TerminalPunctuationFilter`

Filter segments based on a penalty score with respect to the co-occurrence of therminal punctuation marks ('.', '...', '?', '!') in source and target segments. The score is formulated as follows: the initial score is the absolute difference in source and target terminal punctuation counts, the score is then incremented by the number of terminal punctuation beyond the first occurence in both segments, and finally, the score is updated with `score=-log(score+1)` ([Vázquez et al.](https://www.aclweb.org/anthology/W19-5441/)). The score of the greatest co-occurrence is 0 and smaller values indicate greater penalty.

Parameters:

* `threshold`: minimum score threshold (default -2)

The returned score is a single terminal punctuation score. In filtering, the score has to equal to of be greater than the minimum threshold.

#### `NonZeroNumeralsFilter`

Filter segments based on a similarity measure of numerals between the source and target segments with zeros removed. Non-zero numerals are extracted fron both segments preserving the relative order of the numerals. The similarity score between the numeral sequences is produced with `SequenceMatcher.ratio()` from Python's `difflib` library.

Parameters:

* `threshold`: minimum score threshold (default 0.5)

The returned score is a single similarity score. In filtering, the score has to equal to or be greater than the minimum threshold.

### Language model filters

#### `CrossEntropyFilter`

Filter segments by n-gram language model probabilities.

Parameters:

* `src_lm_params`: dictionary for the parameters for the source language model; see below
* `tgt_lm_params`: dictionary for the parameters for the target language model; see below
* `score_type`: select whether to calculate cross-entropy (`entropy`; default), perplixty (`perplexity`) or negative log-probability (`logprob`) scores
* `src_threshold`: upper threshold for source language score when filtering (optional; default 50.0)
* `tgt_threshold`: upper threshold for target language score when filtering (optional; default 50.0)
* `diff_threshold`: upper threshold for absolute difference of source and target language scores when filtering (optional; default 10.0)

Language model paramters for `src_lm_params` and `tgt_lm_params`:

* `filename`: filename for the language model to use
* `arpa`: LM is in ARPA format instead of binary LM (optional; default `true`)
* `unk`: unknown token symbol (optional; default `<UNK>`, case sensitive)
* `include_unks`: include unknown tokens in perplexity calculations (optional; default `false`)
* `ccs`: list of context cues ignored in perplexity calculations (optional; default `null`)
* `mb`: morph boundary marking (optional; default `""`)
* `wb`: word boundary tag (optional; default `"<w>"`)
* `init_hist`: ignore n first tokens after the end-of-sentence tag `</s>` in perplexity calculations (optional; default 2)
* `interpolate`: list of language models (in ARPA format) and interpolation weights (optional; default `null`)

See [train_ngram](#train_ngram) for training the models. Note that the
format, perplexity calculation, and boundary marking options should
match the parameters used in model training; do not change them unless
you know what you are doing.

Separate scores (entropy, perplexity, or negative log-probability) are
returned for the source and target segment. In filtering, the segment
pair is accepted if both values are below the respective thresholds,
and their absolute difference is below the difference threshold.

### Alignment model filters

#### `WordAlignFilter`

Filter segments by word aligment scores.

Parameters:
* `src_tokenizer`: tokenizer for source language (optional; default `null`)
* `tgt_tokenizer`: tokenizer for target language (optional; default `null`)
* `model`: eflomal model type (optional; default 3)
* `priors`: priors for the aligment (optional; default `null`)

The only tokenizer supported at the moment is the
[mosestokenizer](https://github.com/luismsgomes/mosestokenizer) that
wraps the tokenizer script from the Moses toolkit. To enable it,
provide a tuple containing `moses` and an appropriate two-letter
language code, e.g. `[moses, en]` for English.

The eflomal model types are 1 for IBM1, 2 for IBM1 + HMM, and 3 for
IBM1 + HMM + fertility. See https://github.com/robertostling/eflomal
for details.

See [train_aligment](#train_aligment) for training priors. Compatible
tokenizer and model parameters should be used.

## Custom filters

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

Based on the `score` and `accept` methods, the abstract class
`FilterABC` implements the following three generators that take
iterator over segment pairs as input:

* `decisions` yields results of the `accept` method
* `filter` yields only accepted segment pairs
* `filterfalse` yields only rejected segment pairs

These should not be redefined except for a good reason.

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

## Other tools

Apart from the main `opusfilter` script, the packages also provides
the `opusfilter-scores` script. It is a tool that can be used to
calculate and plot statistics from scores produced by the
[`score`](#score) function. The tool has several subcommands that all
take the JSON Lines score file as the input, and either print or plot
the output:

* `list`: Print score column names
* `describe`: Print basic score statistics
* `corr`: Plot score correlation matrix
* `hist`: Plot score histograms
* `scatter-matrix`: Plot scatter matrix for scores
* `values`: Plot score values by line number
