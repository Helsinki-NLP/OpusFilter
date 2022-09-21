# Preprocessing text

## preprocess

Preprocess text with a combination of preprocessors.

Parameters:

* `inputs`: input files for segments to preprocess
* `outputs`: output files for preprocessed segments
* `n_jobs`: number of sub processes to parallel run jobs. If not set, the default value is `default_n_jobs` in `common` section.
* `preprocessors`: a list of preprocessors to apply; see below

The preprocessors parameter is a list of dictionaries, each
representing one preprocessor. The top level should typically include
a single key that defines the class name for the preprocessor
(e.g. `WhitespaceNormalizer`). Additionally it can include a special
key `module` for defining module name for
[custom preprocessors](../preprocessors/custom_preprocessors.md).

Under the class name there is a dictionary the defines the parameters
of the preprocessors. The are mostly specific to the preprocessor
class; see [Available preprocessors](preprocessors) for ready-made preprocessors.
