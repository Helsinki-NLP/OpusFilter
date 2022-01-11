# OpusFilter

OpusFilter is a tool for filtering and combining parallel corpora. It
uses the OpusTools library (Aulamo et al., 2020) to download data from
the OPUS corpus collection (Tiedemann, 2012), but can be used with any
corpora in raw text format.

Features:

* Preprocessing pipelines configured with [YAML](https://yaml.org/)
* Simple downloading of parallel corpora from [OPUS](http://opus.nlpl.eu/) with [OpusTools](https://github.com/Helsinki-NLP/OpusTools)
* Implementations for many common text file operations on parallel files
* Memory-efficient processing of large files
* Implemented filters based e.g. on language identification, word
  aligment, and n-gram language models
* Extendable with your own filters written in Python

OpusFilter has been presented in [ACL 2020 system demonstrations](https://www.aclweb.org/anthology/2020.acl-demos.20).

A changelog is available in [docs/CHANGELOG.md](docs/CHANGELOG.md).

## Table of contents

* [Installing](#installing)
   * [Required libraries](#required-libraries)
   * [Optional libraries and tools](#optional-libraries-and-tools)
* [Citing and references](#citing-and-references)
* [How to contribute](#how-to-contribute)
* [Overview](#overview)
   * [Examples](#examples)
   * [Variables and constants](#variables-and-constants)
   * [Running a single command](#running-a-single-command)
* [Available functions](#available-functions)
   * [Downloading and selecting data](#downloading-and-selecting-data)
      * [opus_read](#opus_read)
      * [concatenate](#concatenate)
      * [download](#download)
      * [head](#head)
      * [tail](#tail)
      * [slice](#slice)
      * [split](#split)
      * [subset](#subset)
      * [product](#product)
      * [unzip](#unzip)
      * [write](#write)
   * [Preprocessing text](#preprocessing-text)
      * [preprocess](#preprocess)
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
      * [AverageWordLengthFilter](#averagewordlengthfilter)
      * [LongWordFilter](#longwordfilter)
   * [Script and language identification filters](#script-and-language-identification-filters)
      * [CharacterScoreFilter](#characterscorefilter)
      * [LanguageIDFilter](#languageidfilter)
   * [Special character and similarity filters](#special-character-and-similarity-filters)
      * [HtmlTagFilter](#htmltagfilter)
      * [TerminalPunctuationFilter](#terminalpunctuationfilter)
      * [NonZeroNumeralsFilter](#nonzeronumeralsfilter)
      * [LongestCommonSubstringFilter](#longestcommonsubstringfilter)
      * [RepetitionFilter](#repetitionfilter)
   * [Language model filters](#language-model-filters)
      * [CrossEntropyFilter](#crossentropyfilter)
      * [CrossEntropyDifferenceFilter](#crossentropydifferencefilter)
      * [LMClassifierFilter](#lmclassifierfilter)
   * [Alignment model filters](#alignment-model-filters)
      * [WordAlignFilter](#wordalignfilter)
* [Custom filters](#custom-filters)
* [Available preprocessors](#available-preprocessors)
   * [Tokenizer](#tokenizer)
   * [Detokenizer](#detokenizer)
   * [WhitespaceNormalizer](#whitespacenormalizer)
   * [RegExpSub](#regexpsub)
   * [MonolingualSentenceSplitter](#monolingualsentencesplitter)
* [Custom preprocessors](#custom-preprocessors)
* [Other tools](#other-tools)
   * [opusfilter-diagram](#opusfilter-diagram)
   * [opusfilter-duplicates](#opusfilter-duplicates)
   * [opusfilter-scores](#opusfilter-scores)
   * [opusfilter-test](#opusfilter-test)

## Installing

Install the latest release from PyPI:
* `pip install opusfilter`

Include optional Python libraries:
* `pip install opusfilter[all]`

Install from source:
* `pip install .` or
* `python setup.py install`

Note that all required libraries are not available to install via PyPI
on Windows OS.  On Linux, it should work directly for Python versions
from 3.6 to 3.8, but with Python 3.9 the `fast-mosestokenizer` library
currently requires a manual install.

### Required libraries

* beautifulsoup4
* graphviz
* langid
* fast-mosestokenizer
* fasttext
* OpusTools
* pandas
* pycld2
* pyhash
* ruamel.yaml
* regex
* scikit-learn
* tqdm

See `setup.py` for possible version requirements.

### Optional libraries and tools

For Chinese tokenization (word segmentation), you can use the
[jieba](https://github.com/fxsjy/jieba) library. It can be installed
automatically with pip by including the extras `[jieba]` or `[all]`
(e.g. `pip install opusfilter[all]`).

For using n-gram language model filters, you need to install VariKN
(https://github.com/vsiivola/variKN) and its Python wrapper. Include
the library files compiled to `build/lib/python` to your Python
library path (e.g. by setting the `PYTHONPATH` environment variable).

For using word alignment filters, you need to install elfomal
(https://github.com/robertostling/eflomal) and set environment
variable `EFLOMAL_PATH` to eflomal's root directory, which contains
the Python scripts `align.py` and `makepriors.py`. Note that you
will need `Cython` to install the Python interface to `eflomal`.

## Citing and references

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

A full bibliography of papers cited in the documentation and code can
be found from [docs/references.bib](docs/references.bib).

## How to contribute

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Overview

The main script provided by the package is `opusfilter`, which takes
a configuration file as an input. The configuration files are in
[YAML](https://yaml.org/) format. At the top level, they have to
sections:

* `common`, which includes global options, and
* `steps`, which is a list of the corpus processing steps.

The syntax for the `opusfilter` is
```
opusfilter [--overwrite] [--last LAST] [--single SINGLE] CONFIG
```
where `CONFIG` is path to the configuration file.
The script will run the steps one by one and stops when the final step
has been processed (if no exceptions were raised). The script has
options for setting the last step to run (`--last`) and running
only a single step (`--single`). It the latter, the user has to
make sure that all input files for the step already exist. The first
step has number 1, and -1 points to the last step, -2 to the second to
last, and so on.

By default, existing output files will be re-used, and the steps
producing them skipped. The `--overwrite` option will force overwrite
for all the steps.

The valid options for the `common` section includes:

* `output_directory` for setting where to write the output files. If
  it is not set, the current working directory is used.
* `chunksize` for changing the default chunk size option for `filter`
  (with `filterfalse` option) and `score` steps. Increasing the value
  from the default 100000 may speed up things at the cost of increased
  memory use.
* `constants` for setting constants; see
  [Variables and constants](#variables-and-constants).

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
      inputs:
      - all.fi.gz
      - all.en.gz
      outputs:
      - filtered.fi.gz
      - filtered.en.gz
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
      inputs:
      - paracrawl.fi.gz
      - paracrawl.en.gz
      outputs:
      - paracrawl_filtered.fi.gz
      - paracrawl_filtered.en.gz
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
      inputs:
      - wmt.fi.gz
      - wmt.en.gz
      outputs:
      - wmt_filtered.fi.gz
      - wmt_filtered.en.gz
      filters: *myfilters
```

### Variables and constants

If you have for example multiple bitexts, for which you want to use
the same preprocessing steps, writing separate steps for each of them
requires a lot of manually written configuration. While generating the
YAML configuration programmatically is a recommended option especially
for very complex setups, OpusFilter provides a couple of custom YAML
tags for defining objects that can be replaced with constant or
variable values.

The first tag, `!var`, is used for mapping the variable name (string)
to its value. For example, if the variable or constant `x` is one in
the current scope, `{key: !var x}` will be replaced by `{key: 1}`.
The value can be any kind of object.

The second tag, `!varstr`, can be used for one or more variable
substitutions within a single string, as in Python's `format()`
method. For example, if `l1` and `l2` have the values `en` and `fi`,
respectively, in the current scope,
`{output: !varstr "cleaned.{l1}-{l2}.txt"}` will be replaced by
`{output: cleaned.en-fi.txt}`.
Note that the string template has to be quoted in order that the YAML
loader will not consider the expressions inside the braces as objects.

Constant values for the variables can be defined in the `common`
section of the configuration, having a scope in all steps, or in
individual steps, having a local scope within the step. In either
case, the definitions are placed in a dictionary under the `constants`
key, with variable names as keys. For example,
```
common:
  constants:
    source: en

steps:
  - type: concatenate
    parameters:
      inputs:
      - !varstr "file1.{source}-{target}.gz"
      - !varstr "file2.{source}-{target}.gz"
      output: !varstr "all.{source}-{target}.gz"
    constants:
      target: fi
```
will produce the output file `"all.en-fi.gz"`. The local scope of the
step overrides definitions in `common`.

Another possibility is to define multiple choices for the variable(s)
within a single step. The definitions are placed in a dictionary under
the `variables` key, with variable names as keys and a list of
intended variable values as values. For example,
```
common:
  constants:
    source: en

steps:
  - type: concatenate
    parameters:
      inputs:
      - !varstr "file1.{source}-{target}.gz"
      - !varstr "file2.{source}-{target}.gz"
      output: !varstr "all.{source}-{target}.gz"
    variables:
      target: [fi, sv]
```
will produce output files `"all.en-fi.gz"` and `"all.en-sv.gz"`. If
values are defined for multiple variables, the list are required to
have the same length. The step is expanded into as many substeps as
there are values in the lists. Note that if you need to use the
same lists of variable values in multiple steps, you can exploit
the standard YAML node anchors and references.

### Running a single command

If you need to run a single OpusFilter function wihtout the need of
writing a complete configuration, the script `opusfilter-cmd` is
provided for convenience. With the command, you can define a
single-step configuration using just command line arguments.

The syntax for the `opusfilter-cmd` is:

```
opusfilter-cmd [--overwrite] [--outputdir OUTDIR] FUNCTION [--parameters PARAMETERS] [PARAMETER-OPTION] ...
```

The `--outputdir` option defines the work directory; if not given the
current directory is used. Existing output files will not be
overwritten by default; the `--overwrite` option will force overwrite.

The `FUNCTION` argument defines the function to use. Parameters for
the function can be defined in two ways: Providing a single JSON
object (as a string) that corresponds to the parameter definition in
YAML configuration, or setting the parameters with custom options.

For the former, use the `--parameters` option. For example, the
following performs filtering with a single word length filter:
```
opusfilter-cmd filter --parameters {"inputs": ["wmt.fi.gz", "wmt.en.gz"], "outputs": ["wmt_filtered.fi.gz", "wmt_filtered.en.gz"], "filters": [{"LengthFilter": {"unit": "word", "min_length": 1, "max_length": 100}}]}
```

Writing a valid complex JSON object may be difficult, and the custom
parameter options help with that. You can replace each top-level key
(e.g. `inputs`) in the parameters with a corresponding option
(`--inputs`) followed by its value, again as a JSON object. A value
that cannot be parsed as JSON will be assumed to be a string. If
multiple values are given for the same option, or the same option is
given multiple times, the value is assumed to be a list containing all
the values. For providing a list of a single value, use the JSON
notation (e.g. `'["value"]'`). Any dashes in the option name will be
replaced by underscores (e.g. `--corpus-name` can be used to produce
the `corpus_name` parameter).

In this manner, the above example can be written also as:
```
opusfilter-cmd filter --inputs wmt.fi.gz wmt.en.gz --outputs wmt_filtered.fi.gz wmt_filtered.en.gz --filters '[{"LengthFilter": {"unit": "word", "min_length": 1, "max_length": 100}}]'
```
Note that the filters still need to be defined as a complex JSON
objects. The created configuration will be shown as YAML for easier
checking and storing.

## Available functions

### Downloading and selecting data

#### `opus_read`

Read a corpus from the OPUS corpus collection (Tiedemann, 2012) using
the OpusTools (Aulamo et al., 2020a) interface.

Parameters:

* `corpus_name`: name of the corpus in OPUS
* `source_language`: language code for the source language
* `target_language`: language code for the target language
* `release`: version of the corpus in OPUS
* `preprocessing`: `raw` for untokenized and `xml` for tokenized segments
* `src_output`: output file for source language
* `tgt_output`: output file for target language
* `suppress_prompts`: `false` (default) prompts user to confirm before download, `true` to download without prompting

#### `concatenate`

Concatenate two or more text files.

Parameters:

* `inputs`: a list of input files
* `output`: output file

#### `download`

Download a file from URL.

Parameters:

* `url`: URL for the file to download
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

* `inputs`: input files
* `outputs`: output files for storing the subset
* `size`: number of lines to select for the subset
* `seed`: seed for the random generator; set to ensure that two runs select the same lines (optional; default `null`)
* `shuffle_subset`: shuffle the order of the selected lines for each language except for the first; can be used to produce noisy examples for training a corpus filtering model (optional; default `false`)

#### `product`

Create a Cartesian product of parallel segments and optionally sample from them.

Parameters:

* `inputs`: a list of input files lists
* `outputs`: a list of output files
* `skip_empty`: skip empty lines (optional, default `true`)
* `skip_duplicates`: skip duplicate lines per language (optional, default `true`)
* `k`: sample at most k random items per product (optional, default `null`)
* `seed`: seed for the random generator; set to ensure that two runs produce the same lines (optional; default `null`)

Can be used to combine parallel files of the same language that
contain alternative translations or other meaningful variation
(e.g. alternative subword segmenatations). For example, if you have
the same text translated to language A by N translators and to
language B by M translators, you can combine the N + M files into two
files having N x M lines for each original line.

#### `unzip`

Unzip parallel segments joined in a single file into multiple files.

Parameters:

* `input`: input file
* `outputs`: a list of output files
* `separator`: a string separator in the input file

Can be used to split e.g. Moses-style (` ||| `) or tab-separated parallel text files into parts.

#### `write`

Write a specified string into a file.

Parameters:

* `output`: output file
* `data`: input data to write to the output (converted to a string if not already)

Useful mostly for testing.

### Preprocessing text

#### `preprocess`

Filter parallel data with a combination of filters.

Parameters:

* `inputs`: input files for segments to preprocess
* `outputs`: output files for preprocessed segments
* `preprocessors`: a list of preprocessors to apply; see below

The preprocessors parameter is a list of dictionaries, each
representing one preprocessor. The top level should typically include
a single key that defines the class name for the preprocessor
(e.g. `WhitespaceNormalizer`). Additionally it can include a special
key `module` for defining module name for [custom preprocessors](#custom-preprocessors).

Under the class name there is a dictionary the defines the parameters
of the preprocessors. The are mostly specific to the preprocessor
class; see section [Available preprocessors](#available-preprocessors)
for ready-made preprocessors.

### Filtering and scoring

#### `remove_duplicates`

Filter out duplicate lines from parallel corpus files.

Parameters:

* `inputs`: input file(s)
* `outputs`: output file(s)
* `compare`: select files for duplicate comparison (optional; default `all` or a list of indices)
* `hash`: select hash algorithm from pyhash (optional; default `xx_64`)
* `overlap`: remove overlap with a second set of files (optional; default `null`)

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

Instead of removing duplicates from a single set, the optional
`overlap` argument can be used to remove all segments from `inputs`
that match the segments in `overlap`. For example, you can remove
exact duplicates of the test data from your training data.

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

* `inputs`: input files for segments to filter
* `outputs`: output files for filtered sentences
* `filters`: a list of filters to apply; see below
* `filterfalse`: yield segment pairs that do not pass at least one of the filters (optional; default `false`)

The filters parameter is a list of dictionaries, each representing one
filter. The top level should typically include a single key that
defines the class name for the filter (e.g. `LenghtFilter`).
Additionally it can include a special key `module` for defining module
name for [custom filters](#custom-filters).

Under the class name there is a dictionary the defines the parameters
of the filters. The are mostly specific to the filter class; see
section [Available filters](#available-filters) for ready-made
filters. An exception is a parameter `name` that is available for all
filters. It has no effect for the filter function, but is useful for
the score function below.

The output of the step is only those segment pairs that are accepted
by all the filters, unless `filterfalse` is set true, in which case
the output is the opposite (i.e., those segment pairs that are
rejected by at least one filter).

#### `score`

Calculate filtering scores for the lines of parallel data.

Parameters:

* `inputs`: input files for segments to score
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
represented in a list or dictionary. Typically there is one score
for each segment (language) if there are multiple scores.

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
(e.g. `x.y` expects that there is key `x` under the key `y`). List items
can be accessed with integer keys. The type conversion can be used e.g.
for forcing numerical values to be compared as strings.

### Training language and alignment models

#### `train_ngram`

Train a character-based varigram language model with VariKN (Siivola et al. 2007).
Can be used for `CrossEntropyFilter` and `CrossEntropyDifferenceFilter`.

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

Train word alignment priors for eflomal (Östling and Tiedemann, 2016).
Can be used in `WordAlignFilter`.

Parameters:

* `src_data`: input file for the source language
* `tgt_data`: input file for the target language
* `parameters`: training options for the aligment and tokenization
   * `src_tokenizer`: tokenizer for source language (optional; default `null`)
   * `tgt_tokenizer`: tokenizer for target language (optional; default `null`)
   * `model`: eflomal model type (optional; default 3)
   * `scores`: file to write alignment scores from the training data (optional; default `null`)
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
* `pass_empty`: if `true`, always accept if all segment lengths are zero (default `false`)

Returned scores are lengths for the source and target segment. In
filtering, all segments have to be between the minimum and maximum
length thresholds.

#### `LengthRatioFilter`

Filtering based on ratio of the segment lengths.

Parameters:

* `threshold`: threshold for the length ratio
* `unit`: type of unit for calculating the lengths (optional; `word` for words or any whitespace-separated units, and `character` or `char` for characters; the default is `word`)

Returned score is the higher length divided by the lower length, or
infinity of either of the lengths are zero. In filtering, segment
pairs is accepted of the ratio is below the given threshold.

#### `AverageWordLengthFilter`

Filtering based on average word lengths.

Parameters:

* `min_length`: minimum length (optional; default 2)
* `max_length`: maximum length (optional; default 20)
* `pass_empty`: if `true`, always accept if all segment lengths are zero (default `false`)

Returned scores are average words lengths for the segments. In
filtering, all segments have to be between the minimum and maximum
length thresholds.

#### `LongWordFilter`

Filtering based on maximum word length.

Parameters:

* `threshold`: maximum length (optional; default 40)

Returned score is the length of the longests words across the
segments. The length has to below the threshold.

### Script and language identification filters

#### `CharacterScoreFilter`

Filter segments based on what proportion of their alphabetic characters are in a given script. For a list of valid scripts, see e.g. https://www.regular-expressions.info/unicode.html

Parameters:

* `scripts`: scripts for input segments
* `thresholds`: minimum proportion of characters in a script (default 1)

Returned scores are proportions of valid characters in the segments. In filtering, all values have to be equal to or greater than the minimum thresholds.

#### `LanguageIDFilter`

Filter segments based on their language identification confidence scores.

Parameters:

* `languages`: expected languages (ISO639 language codes) for the segments
* `id_method`: language indentification method (`langid` for using the `langid` library, `cld2` for using the `cld2` library, or `fasttext` for using a `fasttext` model; the default is `langid`)
* `thresholds`: minimum identification confidence score for the segments (a single float or a list of floats per language)
* `fasttext_model_path`: path for a `fasttext` model (required only for the `fasttext` method; default `null`)
* `langid_languages`: limit detection to a list of possible languages (valid only for the `langid` method; default `null`)
* `cld2_options`: a dictionary of options for the `cld2` method (valid only for the `cld2` method; default `null`)

Returned scores are the language identification confidence scores from a given identification method for the segments. The scores range from 0 to 1. In filtering, all values have to be greater than the minimum thresholds. Negative threshold can be used to skip filtering for a language.

See [langid.py](https://github.com/saffsd/langid.py) and [pycld2](https://github.com/aboSamoor/pycld2) for the method-specific options.

A pretrained `fasttext` model can be downloaded from https://fasttext.cc/docs/en/language-identification.html

### Special character and similarity filters

#### `HtmlTagFilter`

Filter segments based on whether they contain HTML tags or not.

The returned scores are two boolean values indicating whether the segments contain HTML tags. In filtering, a segment pair is accepted if none of the segments contains HTML tags.

#### `TerminalPunctuationFilter`

Filter segments based on a penalty score with respect to the co-occurrence of therminal punctuation marks ('.', '...', '?', '!') in source and target segments (Vázquez et al., 2019). The score is formulated as follows: the initial score is the absolute difference in source and target terminal punctuation counts, the score is then incremented by the number of terminal punctuation beyond the first occurence in both segments, and finally, the score is updated with `score=-log(score+1)`. The score of the greatest co-occurrence is 0 and smaller values indicate greater penalty.

This filter works only for bilingual input.

Parameters:

* `threshold`: minimum score threshold (default -2)

The returned score is a single terminal punctuation score. In filtering, the score has to equal to of be greater than the minimum threshold.

#### `NonZeroNumeralsFilter`

Filter segments based on a similarity measure of numerals between the segments with zeros removed (Vázquez et al., 2019). Non-zero numerals are extracted from all segments preserving the relative order of the numerals. The similarity score between the numeral sequences is produced with `SequenceMatcher.ratio()` from Python's `difflib` library.

Parameters:

* `threshold`: minimum score threshold (default 0.5)
* `require_all`: if True, all scores (for pairs of n segments) have to be reach threshold; otherwise at least one the ratios has to reach the threshold

The returned value is a list of similarity scores for all language pairs. For n-lingual input, the scores will include C(n, 2) values. In filtering, all pairwise scores has to equal to or be greater than the minimum threshold.

#### `LongestCommonSubstringFilter`

Filter segments based on similarity of the strings.

Parameters:

* `threshold`: filter segments if the similarity is equal or above the threshold (optional; default 0.9)
* `require_all`: if True, all ratios (for pairs of n segments) have to be below the threshold; otherwise at least one the ratios have to be below the threshold

Returned scores are ratios between the length of the longest common substring and the length of the shorter of the compared strings for all language pairs. For n-lingual input, the scores will include C(n, 2) values.

#### `RepetitionFilter`

Filter segments with repeated content. Useful e.g. for filtering data generated by a low-quality NMT model.

Parameters:

* `threshold`: number of repetitions required to activate the filter (optional, default 2)
* `min_length`: minimum number of characters in the repeated sequence (optional, default 3)
* `max_length`: maximum number of characters in the repeated sequence (optional, default 100)

The returned scores are the numbers of repetitions if at least
threshold repetitions were found (first occurrence of the string is
not counted), or zero if no repetitions were found, or all were below
the threshold. The returned number of repetitions is for the first
match, and it is possible that the segment contains longer
repetitions.

There may be optional space character(s) between the repeated strings
that are not counted to the length. The repeated string cannot start
with a whitespace character but is not limited otherwise.

### Language model filters

#### `CrossEntropyFilter`

Filter segments by n-gram language model probabilities.

Parameters:

* `lm_params`: a list of dictionaries for the parameters of the language models; see below
* `score_type`: select whether to calculate cross-entropy (`entropy`; default), perplixty (`perplexity`) or negative log-probability (`logprob`) scores
* `thresholds`: upper thresholds for scores when filtering (optional; default is 50.0 for all languages)
* `low_thresholds`: lower thresholds for scores when filtering (optional; default is no threshold)
* `diff_threshold`: upper threshold for absolute difference of source and target language scores when filtering (optional; default 10.0)
* `score_for_empty`: set score values manually for empty input pairs (default `null`)

Language model parameters for `lm_params`:

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
pair is accepted if all values are below the respective thresholds,
and their absolute differences are below the difference threshold.

#### `CrossEntropyDifferenceFilter`

Filter segments by using cross-entropy difference method by Moore and Lewis (2010).

Parameters:

* `id_lm_params`: a list of dictionaries for the parameters of the in-domain language models
* `nd_lm_params`: a list of dictionaries for the parameters of the non-domain language models
* `thresholds`: upper thresholds for scores when filtering (optional; default is 0.0 for all languages)
* `score_for_empty`: set score values manually for empty input pairs (default `null`)

For contents of the `id_lm_params` and `nd_lm_params`, see
[CrossEntropyFilter](#crossentropyfilter).

See [train_ngram](#train_ngram) for training the models. Note that the
format, perplexity calculation, and boundary marking options should
match the parameters used in model training; do not change them unless
you know what you are doing.

The filter returns difference between the in-domain LM and non-domain
LM cross-entropy.

#### `LMClassifierFilter`

Filter segments by using classification probability from a naive Bayes
classifier using a set class-specific language models.

Parameters:

* `labels`: expected class labels for the segments
* `lm_params`: a dictionary that maps labels to language model parameter dictionaries
* `thresholds`: minimum thresholds for probability of the expected label when filtering (optional; default is 0.5)
* `relative_score`: normalize probabilities by the largest probability (optional; default `false`)

Each of the labels should have a corresponding language model in
`lm_params`. The likelihood of the segment is calculated for all the
language models. If `relative_score` is false, the likelihoods are
normalized to probabilities that sum up to one over the labels, and
the probability of the expected label is returned as a score. If
`relative_score` is true, the probability values are first divided by
the largest probability (i.e. one of the labels will always get one as
the score).

For contents of the language model parameters in `lm_params`, see
[CrossEntropyFilter](#crossentropyfilter).

See [train_ngram](#train_ngram) for training the models. Note that the
format, perplexity calculation, and boundary marking options should
match the parameters used in model training; do not change them unless
you know what you are doing.

A possible use case for this filter is creating a custom language
identifier similar to Vatanen et al. (2010): Train a character-based
n-gram model for each of the languages from clean corpora, and use the
language codes as labels. Vatanen et al. (2010) recommend using
absolute discounting and maximum n-gram length 4 or 5 for the
models. Note that unknown tokens are ignored in the language model
likelihoods, so it is a good idea to train a small (e.g. unigram)
background model that includes data from all languages, and
interpolate the language-specific models with it using a small
interpolation coefficient. An example configuration is found at
[example_configs/qed_lm_langid.yaml](example_configs/qed_lm_langid.yaml).

### Alignment model filters

#### `WordAlignFilter`

Filter segments by word aligment scores using eflomal (Östling and Tiedemann, 2016).

Parameters:
* `src_threshold`: score threshold for source language (default 0)
* `tgt_threshold`: score threshold for target language (default 0)
* `src_tokenizer`: tokenizer for source language (optional; default `null`)
* `tgt_tokenizer`: tokenizer for target language (optional; default `null`)
* `model`: eflomal model type (optional; default 3)
* `priors`: priors for the aligment (optional; default `null`)
* `score_for_empty`: score values for empty input pairs (default -100)

A segment pair is accepted if scores for both directions are lower
than the corresponding thresholds.

The supported tokenizers are listed in [Tokenizer](#tokenizer). For
example, to enable Moses tokenizer, provide a tuple containing `moses`
and an appropriate two-letter language code, e.g. `[moses, en]` for
English.

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
should be called with the remaining keyword arguments. The keyword
argument `name` is reserved for giving names to the filters and
`workdir` for a location for non-temprary files.

Based on the `score` and `accept` methods, the abstract class
`FilterABC` implements the following three generators that take
iterator over segment pairs as input:

* `decisions` yields results of the `accept` method
* `filter` yields only accepted segments
* `filterfalse` yields only rejected segments

These should not be redefined except for a good reason.

The example below shows code for simple filter that calculates the
proportion of uppercase letters in the sentences, and accepts the pair
only if all sentences have less than 50% (or given threshold) of
uppercase characters:

```python
import opusfilter

class UppercaseFilter(opusfilter.FilterABC):

    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def uppercase_ratio(self, sentence):
        length = len(sentence)
        if length > 0:
            return sum(1 for char in sent if char.isupper()) / length
        return 0

    def score(self, pairs):
        for pair in pairs:
            yield [self.uppercase_ratio(sentence) for sentence in pair]

    def accept(self, score):
        return all(ratio < self.threshold for ratio in score)
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

If a filter requires external resources files (e.g. for model
parameters), or stores non-temporary files itself, they should be
located in the path defined the attribute `workdir`. The
implementation of the filter should join `workdir` with relative file
paths using `os.path.join()`.

## Available preprocessors

### `Tokenizer`

Tokenize parallel texts.

Parameters:

* `tokenizer`: tokenizer type or a list of types for each input
* `languages`: a list of language codes for each input
* `options`: tokenizer options dictionary or a list of tokenizer dictionaries for multiple tokenziers (optional)

Supported tokenizers:

* `moses`:
  * Uses the [fast-mosestokenizer](https://github.com/mingruimingrui/fast-mosestokenizer) package.
  * Avaliable for most languages.
  * Options are passed to the `mosestokenizer.MosesTokenizer` class; see its documentation for the available options.
* `jieba`:
  * Uses the [jieba](https://github.com/fxsjy/jieba) package.
  * Only avaliable for Chinese (zh, zh_CN).
  * Options are passed to `jieba.cut` function; see its documentation for the avaliable options.
  * If you use `jieba`, please install OpusFilter with extras `[jieba]` or `[all]`.

The list of language codes should match to the languages of the input
files given in the `preprocess` step. If more than on tokenizer is
provided, the length of the list should match the number of the
languages. If more than one tokenizer options are provided, the length
should again match the number of the languages.

### `Detokenizer`

Detokenize parallel texts.

Parameters:

* `tokenizer`: tokenizer type or a list of types for each input
* `languages`: a list of language codes for each input
* `options`: tokenizer options dictionary or a list of tokenizer dictionaries for multiple tokenziers (optional)

See [Tokenizer](#tokenizer) for description of the parameters.

### `WhitespaceNormalizer`

Normalizes whitespace characters.

The whitespace normalization consists of two steps: First, any
sequence of one or more unicode whitespace characters (as matched by
`\s` in the `re` standard library) are replaced by a single space
character. Second, any leading and trailing whitespace is removed.

### `RegExpSub`

Run a sequence of arbitrary regular expression substitutions.

Parameters:

* `patterns`: a list of patterns to use by default
* `lang_patterns`: a dictionary or list of patterns to use for specific languages

Multiple substitutions are applied in the given order. The default
patterns are replaced with language-specific patterns when the
corresponding index (starting from 0) is found in the `lang_patterns`
dictionary. The `lang_patterns` argument may also be a list, if you
e.g. want to use separate patterns for all languages.

The substitution patterns are 4-tuples containing the regular
expression, replacement, count (0 = substitute all) and flags (list of
flag constants in the `re` library, e.g. `["I", "A"]`). The regular
expressions are first compiled with [`re.compile`](https://docs.python.org/3/library/re.html#re.compile),
and then the substitutions are applied with [`re.sub`](https://docs.python.org/3/library/re.html#re.sub).

### `MonolingualSentenceSplitter`

Split monolingual text segments into sentences.

Parameters:

* `language`: language code for the input
* `non_breaking_prefix_file`: override the language's non-breaking prefix file by a custom one (optional; default `null`)
* `enable_parallel`: do not raise expection if the input is parallel data (optional; default `false`)

Sentence splitting method imported from the
[sentence-splitter](https://github.com/mediacloud/sentence-splitter)
library. Uses a heuristic algorithm by Philipp Koehn and Josh
Schroeder developed for the Europarl corpus (Koehn, 2005). Supports
mostly European languages, but a non-breaking prefix file for new
languages can be provided.

Warning: This is not intended for parallel data, as there the number
of output lines per each parallel input line would not always
match. Because of this, you can define only a single language, and an
exception is raised if multiple input files are provided. The
exception can be disabled with the `enable_parallel` option for
special cases.

## Custom preprocessors

Similarly to filters, You can import your own preprocessors by
defining the `module` key in the filter configuration entries.

The custom preprocessors should inherit the abstract base class
`PreprocessorABC` from the `opusfilter` package, and implement the
abstract methods `process`. The `process` method is a generator that
takes an iterator over segments and yields preprocessed (modified)
segments of text. It also has an additional argument, `f_idx`, which
is the index of the current file being processed by the `preprocess`
step. This argument enables preprocessing parallel files in the same
step even if the preprocessing options (such as language code for a
tokenizer) varies.

## Other tools

Apart from the main [`opusfilter`](#overview) and [`opusfilter-cmd`](#running-a-single-command)
scripts, the package also provides some analysis tools.

### `opusfilter-diagram`

Draws a diagram (a directed acyclic graph) from OpusFilter
configuration file using the `graphviz` library.

```
opusfilter-diagram [--rankdir {TB,LR}] FILE FILE
```

The `--rankdir` option changes the direction of the graph from
left-to-right (default) to top-to-bottom. If the output file ends
with `.dot`, the raw dot format is used; otherwise the graph is
rendered to the format indicated by the extension (e.g. PDF or PNG).

### `opusfilter-duplicates`

This is a simple script based on the [`remove_duplicates`](#remove_duplicates)
function, that instead of filtering the data, prints out statistics of
the duplicate entries. You can either provide a single corpus (as one
monolingual file or multiple parallel files) for calculating the
number of duplicates in it, or two corpora for calculating the overlap
between them. The syntax for the `opusfilter-duplicates` is:

```
opusfilter-duplicates [--overlap FILE [FILE ...]] [--hash HASH] [--letters-only] [--lowercase] FILE [FILE ...]
```

The options are essentially the same as for [`remove_duplicates`](#remove_duplicates).

### `opusfilter-scores`

This is a tool that can be used to calculate and plot statistics from
scores produced by the [`score`](#score) function. The tool has
several subcommands that all take the JSON Lines score file as the
input, and either print or plot the output:

* `list`: Print score column names
* `describe`: Print basic score statistics
* `corr`: Plot score correlation matrix
* `hist`: Plot score histograms
* `scatter-matrix`: Plot scatter matrix for scores
* `values`: Plot score values by line number

### `opusfilter-test`

This is a simple script based on the [`filter`](#filter) function that
can be used to calculate the amount of segments that the given
filter(s) would remove from the parallel data, and optionally output
the to-be-removed segments.

The syntax for the `opusfilter-test` is:

```
opusfilter-test [--yaml FILE] [--add CLASS JSON] [--removed FILE] FILE [FILE ...]
```

The filters to test can be defined either from a YAML file (`--yaml`)
using a similar definition as the `filters` parameter for the `filter`
function, or adding the one by one with the `--add` option, which
takes the filter class as the first argument and filter parameters in
as a JSON object as the second argument. For default filter
parameters, an empty dictionary (`'{}'`) should be provided.

The scripts first calculates the total number of segments in the input
files, and then runs the filters on them one by one. The number and
proportion of removed segments is printed. In addition, it is possible
to write the removed segments to a file in JSON Lines format
(`--removed`).
