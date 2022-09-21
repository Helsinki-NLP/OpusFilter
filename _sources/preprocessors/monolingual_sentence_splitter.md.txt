# MonolingualSentenceSplitter

Split monolingual text segments into sentences.

Parameters:

* `language`: language code for the input
* `non_breaking_prefix_file`: override the language's non-breaking prefix file by a custom one (optional; default `null`)
* `enable_parallel`: do not raise expection if the input is parallel data (optional; default `false`)

Sentence splitting method imported from the
[sentence-splitter](https://github.com/mediacloud/sentence-splitter)
library. Uses a heuristic algorithm by Philipp Koehn and Josh
Schroeder developed for the Europarl corpus {cite:p}`koehn-2005-europarl`.
Supports mostly European languages, but a non-breaking prefix file for
new languages can be provided.

Warning: This is not intended for parallel data, as there the number
of output lines per each parallel input line would not always
match. Because of this, you can define only a single language, and an
exception is raised if multiple input files are provided. The
exception can be disabled with the `enable_parallel` option for
special cases.
