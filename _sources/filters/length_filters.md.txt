# Length filters

## LengthFilter

Filtering based on absolute segment lengths.

Parameters:

* `min_length`: minimum segment length (optional; default 1)
* `max_length`: maximum segment length (optional; default 100)
* `unit`: type of unit for calculating the lengths (optional; `word` for words or any whitespace-separated units, and `character` or `char` for characters; the default is `word`)
* `pass_empty`: if `true`, always accept if all segment lengths are zero (default `false`)

Returned scores are lengths for the source and target segment. In
filtering, all segments have to be between the minimum and maximum
length thresholds.

Any of the `min_length`, `max_length`, and `unit` parameters can also
be given as lists, in which case Nth entry in the list is applied to
the Nth of the parallel input segments.

## LengthRatioFilter

Filtering based on ratio of the segment lengths.

Parameters:

* `threshold`: threshold for the length ratio
* `unit`: type of unit for calculating the lengths (optional; `word` for words or any whitespace-separated units, and `character` or `char` for characters; the default is `word`)

Returned score is the higher length divided by the lower length, or
infinity of either of the lengths are zero. In filtering, segment
pairs is accepted of the ratio is below the given threshold.

In order to use different units per language, the `unit` parameter can
also be given as a list.

## AverageWordLengthFilter

Filtering based on average word lengths.

Parameters:

* `min_length`: minimum length (optional; default 2)
* `max_length`: maximum length (optional; default 20)
* `pass_empty`: if `true`, always accept if all segment lengths are zero (default `false`)

Returned scores are average words lengths for the segments. In
filtering, all segments have to be between the minimum and maximum
length thresholds.

The `min_length` and `max_length` parameters can also be given as
lists, in which case Nth entry in the list is applied to the Nth of
the parallel input segments.

## LongWordFilter

Filtering based on maximum word length.

Parameters:

* `threshold`: maximum length (optional; default 40)

Returned score is the length of the longests words across the
segments. The length has to below the threshold.

In order to allow have different thresholds per language, the
threshold parameter can also be given as a list.
