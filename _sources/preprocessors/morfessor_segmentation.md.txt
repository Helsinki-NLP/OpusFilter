# MorfessorSegmentation

Split words into subword units using Morfessor model
{cite:p}`virpioja-etal-2013-morfessor`.

* `model`: a Morfessor binary model file
* `separator`: (optional; default `"@@ "`)
* `lowercase`: lowercase all text prior to segmentation (optional; default `false`)
* `viterbi_max_len`: (optional; default 30)
* `viterbi_smoothing`: (optional; default 0)

See [train_morfessor](train_morfessor) for training a model and 
[Morfessor 2.0 documentation](https://morfessor.readthedocs.io/en/latest/)
for details of the parameters.
