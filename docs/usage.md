# Basic usage

## opusfilter script

The main script provided by the package is `opusfilter`, which takes
a configuration file as an input. The configuration files are in
[YAML](https://yaml.org/) format. At the top level, they have to
sections:

* `common`, which includes global options, and
* `steps`, which is a list of the corpus processing steps.

The syntax for the `opusfilter` command is
```
opusfilter [--overwrite] [--last LAST] [--single SINGLE] [--n-jobs N_JOBS] CONFIG
```
where `CONFIG` is path to the configuration file.
The script will run the steps one by one and stops when the final step
has been processed (if no exceptions were raised). The script has
options for setting the last step to run (`--last`) and running
only a single step (`--single`). It the latter, the user has to
make sure that all input files for the step already exist. The first
step has number 1, and -1 points to the last step, -2 to the second to
last, and so on. The `--n-jobs` option indicate number of processes to
use when running `score`, `filter` and `preprocess` steps. This value will 
overwrite `default_n_jobs` in the `common` section.

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
* `default_n_jobs` for defining the number of parallel processes to use
  for `score`, `filter`, and `preprocess` steps. The default value is 1.
* `constants` for setting constants; see
  [Variables and constants](#variables-and-constants).

Each step in `steps` is a dictionary (mapping) with two keys: `type`
and `parameters`. Type is a string that defines the function that
should be run, and parameters is a dictionary with keys that depend on
the function; at minimum the output files are defined there.

## Configuration examples

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

## Variables and constants

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
```yaml
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
```yaml
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

## Running a single command

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
