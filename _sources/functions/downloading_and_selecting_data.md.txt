# Downloading and selecting data

## opus_read

Read a corpus from the OPUS corpus collection {cite:p}`tiedemann-2012-parallel` using
the OpusTools {cite:p}`aulamo-etal-2020-opustools` interface.

Parameters:

* `corpus_name`: name of the corpus in OPUS
* `source_language`: language code for the source language
* `target_language`: language code for the target language
* `release`: version of the corpus in OPUS
* `preprocessing`: `moses` or `raw` for untokenized and `xml` for tokenized segments
* `src_output`: output file for source language
* `tgt_output`: output file for target language
* `suppress_prompts`: `false` (default) prompts user to confirm before download, `true` to download without prompting

The `moses` preprocessing type (available with `OpusTools` version
1.6.2 and above) is recommended for those corpora for which it
exists. The output is equivalent to `raw`, but in some cases it can
significantly reduce the amount of data downloaded in the process.

## concatenate

Concatenate two or more text files.

Parameters:

* `inputs`: a list of input files
* `output`: output file

## download

Download a file from URL.

Parameters:

* `url`: URL for the file to download
* `output`: output file

## head

Take the first n lines from files.

Parameters:

* `inputs`: a list of input files
* `outputs`: a list of output files
* `n`: number of output lines

## tail

Take the last n lines from files.

Parameters:

* `inputs`: a list of input files
* `outputs`: a list of output files
* `n`: number of output lines

Note: The memory requirement of `tail` is proportional to n. Use
`slice` if you need all except the first n lines.

## slice

Take slice of lines from files.

Parameters:

* `inputs`: a list of input files
* `outputs`: a list of output files
* `start`: start index (optional; default 0)
* `stop`: stop index (optional; default `null`)
* `step`: step size (optional; default 1)

Either `start`, `stop`, or both of them should be given. If `stop` is
not given, reads until the end of the file.

## split

Split files to two parts giving the approximative proportions as
fractions.

Parameters:

* `inputs`: input file(s)
* `outputs`: output file(s) for selected lines
* `outputs_2`: output file(s) for the rest of the lines (optional)
* `divisor`: divisor for the modulo operation (e.g. 2 for splitting to equal sized parts)
* `threshold`: threshold for the output of the modulo operation (optional; default 1)
* `compare`: select files to use for hash operation (optional; default `all` or a list of indices)
* `hash`: select hash algorithm from xxhash (optional; default `xxh64`)
* `seed`: integer seed for the hash algorithm (optional; default 0)

Input files are processed line by line in parallel. If the condition
`hash(content) % divisor < threshold`, where the content is a
concatenation of the input lines and the hash function returns an
integer, holds, the lines are written to the `outputs`. If the
condition does not hold, and `outputs_2` are defined, the lines are
written there.

Compared to random splitting (see [subset](subset)) or using the
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

## subset

Take a random subset from parallel corpus files.

Parameters:

* `inputs`: input files
* `outputs`: output files for storing the subset
* `size`: number of lines to select for the subset
* `seed`: seed for the random generator; set to ensure that two runs select the same lines (optional; default `null`)
* `shuffle_subset`: shuffle the order of the selected lines for each language except for the first; can be used to produce noisy examples for training a corpus filtering model (optional; default `false`)

## product

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

## unzip

Unzip parallel segments joined in a single file into multiple files.

Parameters:

* `input`: input file
* `outputs`: a list of output files
* `separator`: a string separator in the input file

Can be used to split e.g. Moses-style (` ||| `) or tab-separated parallel text files into parts.

## write

Write a specified string into a file.

Parameters:

* `output`: output file
* `data`: input data to write to the output (converted to a string if not already)

Useful mostly for testing.
