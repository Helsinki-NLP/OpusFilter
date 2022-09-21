# BPESegmentation

Split words into subword units using BPE model
{cite:p}`sennrich-etal-2016-neural`.

Parameters:

* `model`: a model file with BPE codes
* `merges`: use this many BPE operations (optional; default -1 = use all learned operations)
* `separator` separator between non-final subword units (optional; default `"@@"`)
* `vocab`: vocabulary file; if provided, reverts any merge operations that produce an OOV (optional; default `null`)
* `glossaries`: words matching any of the words/regex provided in glossaries will not be affected (optional; default `null`)
* `dropout`: dropout BPE merge operations with the probability (optional; default 0)

See [train_bpe](train_bpe) for training a model and [subword-nmt documentation](https://github.com/rsennrich/subword-nmt) for details of the parameters.
