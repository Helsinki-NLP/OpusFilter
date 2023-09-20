# Command line tools for analysis

Apart from the main scripts ([`opusfilter`](usage.md#opusfilter-script) and
[`opusfilter-cmd`](usage.md#running-a-single-command)) and configuration
generation ([`opusfilter-autogen`](automatic_configuration.md)),
the package also provides command line tools for testing and analyzing
the configurations and filters.

## opusfilter-diagram

Draws a diagram (a directed acyclic graph) from OpusFilter
configuration file using the `graphviz` library.

```
opusfilter-diagram [--rankdir {TB,LR}] FILE FILE
```

The `--rankdir` option changes the direction of the graph from
left-to-right (default) to top-to-bottom. If the output file ends
with `.dot`, the raw dot format is used; otherwise the graph is
rendered to the format indicated by the extension (e.g. PDF or PNG).

## opusfilter-duplicates

This is a simple script based on the [`remove_duplicates`](remove_duplicates)
function, that instead of filtering the data, prints out statistics of
the duplicate entries. You can either provide a single corpus (as one
monolingual file or multiple parallel files) for calculating the
number of duplicates in it, or two corpora for calculating the overlap
between them. The syntax for the `opusfilter-duplicates` is:

```
opusfilter-duplicates [--overlap FILE [FILE ...]] [--hash HASH] [--letters-only] [--lowercase] FILE [FILE ...]
```

The options are essentially the same as for [`remove_duplicates`](remove_duplicates).

## opusfilter-scores

This is a tool that can be used to calculate and plot statistics from
scores produced by the [`score`](score) function. The tool has
several subcommands that all take the JSON Lines score file as the
input, and either print or plot the output:

* `list`: Print score column names
* `describe`: Print basic score statistics
* `corr`: Plot score correlation matrix
* `hist`: Plot score histograms
* `scatter-matrix`: Plot scatter matrix for scores
* `values`: Plot score values by line number

## opusfilter-test

This is a simple script based on the [`filter`](filter) function that
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
(`--removed`), and collect the scores from the filters to similarly to
the score function (`--scores`).
