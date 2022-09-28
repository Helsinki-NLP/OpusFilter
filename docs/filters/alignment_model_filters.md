# Alignment model filters

## WordAlignFilter

Filter segments by word aligment scores using eflomal
{cite:p}`ostling-tiedemann-2016-efficient`.

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

The supported tokenizers are listed in [Tokenizer](Tokenizer). For
example, to enable Moses tokenizer, provide a tuple containing `moses`
and an appropriate two-letter language code, e.g. `[moses, en]` for
English.

The eflomal model types are 1 for IBM1, 2 for IBM1 + HMM, and 3 for
IBM1 + HMM + fertility. See [github.com/robertostling/eflomal](https://github.com/robertostling/eflomal)
for details.

See [train_aligment](train_aligment) for training priors. Compatible
tokenizer and model parameters should be used.

**Caveats:**

`eflomal` is a stochastic method and two runs will not produce exactly
the same result. Thus, if you include it in your pipeline, full
replicability of the end result is not possible.

Moreover, `eflomal` estimates the model parameters using the input
data, even if you provide the priors file. In consequence, the size of
the input matters, and aligning the same data in chunks will not
provide the same results as aligning all at once. Thus the following
*may* give worse results than expected:

* Using `score` if input size is larger than `chunksize` (default 100000)
* Using `filter` with `filterfalse` on if input size is larger than `chunksize` (default 100000)
* Using parallel processing (`n_jobs` > 1) for `score` or `filter` regardless of the other options
