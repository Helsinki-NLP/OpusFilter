# Filtering and scoring

## remove_duplicates

Filter out duplicate lines from parallel corpus files.

Parameters:

* `inputs`: input file(s)
* `outputs`: output file(s)
* `compare`: select files for duplicate comparison (optional; default `all` or a list of indices)
* `hash`: select hash algorithm from xxhash (optional; default `xxh64`)
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
the result to produce the final key for storing the counts. The
default 64-bit XXHash algorithm should be fine for any practical data
sizes, but if do not care about memory use and want to be extra sure
there are no collisions, you can disable hashing by setting the `hash`
parameter as empty string or null.

## filter

Filter parallel data with a combination of filters.

Parameters:

* `inputs`: input files for segments to filter
* `outputs`: output files for filtered sentences
* `n_jobs`: number of sub processes to parallel run jobs. If not set, the default value is `default_n_jobs` in `common` section.
* `filters`: a list of filters to apply; see below
* `filterfalse`: yield segment pairs that do not pass at least one of the filters (optional; default `false`)

The filters parameter is a list of dictionaries, each representing one
filter. The top level should typically include a single key that
defines the class name for the filter (e.g. `LenghtFilter`).
Additionally it can include a special key `module` for defining module
name for [custom filters](../filters/custom_filters.md).

Under the class name there is a dictionary the defines the parameters
of the filters. The are mostly specific to the filter class; see
[Available filters](filters) for ready-made
filters. An exception is a parameter `name` that is available for all
filters. It has no effect for the filter function, but is useful for
the score function below.

The output of the step is only those segment pairs that are accepted
by all the filters, unless `filterfalse` is set true, in which case
the output is the opposite (i.e., those segment pairs that are
rejected by at least one filter).

## score

Calculate filtering scores for the lines of parallel data.

Parameters:

* `inputs`: input files for segments to score
* `output`: output file for the scores
* `n_jobs`: number of sub processes to parallel run jobs. If not set, the default value is `default_n_jobs` in `common` section.
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
scores or training a classifier for filtering. The JSON Lines data is
easy to load as a [pandas](https://pandas.pydata.org/) DataFrame using
the [`json_normalize`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.json.json_normalize.html)
method.
