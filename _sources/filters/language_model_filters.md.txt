# Language model filters

## CrossEntropyFilter

Filter segments by n-gram language model probabilities.

Parameters:

* `lm_params`: a list of dictionaries for the parameters of the language models; see below
* `score_type`: select whether to calculate cross-entropy (`entropy`; default), perplixty (`perplexity`) or negative log-probability (`logprob`) scores
* `thresholds`: upper thresholds for scores when filtering (optional; default is 50.0 for all languages)
* `low_thresholds`: lower thresholds for scores when filtering (optional; default is no threshold)
* `diff_threshold`: upper threshold for absolute difference of source and target language scores when filtering (optional; default 10.0)
* `score_for_empty`: set score values manually for empty input pairs (default `null`)

Language model parameters for `lm_params`:

* `filename`: filename for the language model to use
* `arpa`: LM is in ARPA format instead of binary LM (optional; default `true`)
* `unk`: unknown token symbol (optional; default `<UNK>`, case sensitive)
* `include_unks`: include unknown tokens in perplexity calculations (optional; default `false`)
* `ccs`: list of context cues ignored in perplexity calculations (optional; default `null`)
* `segmentation`: subword segmentation options (optional; default `{}`)
* `mb`: morph boundary marking (optional; default `""`)
* `wb`: word boundary tag (optional; default `"<w>"`)
* `init_hist`: ignore n first tokens after the end-of-sentence tag `</s>` in perplexity calculations (optional; default 2)
* `interpolate`: list of language models (in ARPA format) and interpolation weights (optional; default `null`)

See [train_ngram](train_ngram) for training the models. Note that the
format, perplexity calculation, segmentation, and boundary marking
options should match the parameters used in model training; do not
change them unless you know what you are doing.

Separate scores (entropy, perplexity, or negative log-probability) are
returned for the source and target segment. In filtering, the segment
pair is accepted if all values are below the respective thresholds,
and their absolute differences are below the difference threshold.

## CrossEntropyDifferenceFilter

Filter segments by using cross-entropy difference method by
{cite:t}`moore-lewis-2010-intelligent`.

Parameters:

* `id_lm_params`: a list of dictionaries for the parameters of the in-domain language models
* `nd_lm_params`: a list of dictionaries for the parameters of the non-domain language models
* `thresholds`: upper thresholds for scores when filtering (optional; default is 0.0 for all languages)
* `score_for_empty`: set score values manually for empty input pairs (default `null`)

For contents of the `id_lm_params` and `nd_lm_params`, see
[CrossEntropyFilter](CrossEntropyFilter).

See [train_ngram](train_ngram) for training the models. Note that the
format, perplexity calculation, and boundary marking options should
match the parameters used in model training; do not change them unless
you know what you are doing.

The filter returns difference between the in-domain LM and non-domain
LM cross-entropy.

## LMClassifierFilter

Filter segments by using classification probability from a naive Bayes
classifier using a set class-specific language models.

Parameters:

* `labels`: expected class labels for the segments
* `lm_params`: a dictionary that maps labels to language model parameter dictionaries
* `thresholds`: minimum thresholds for probability of the expected label when filtering (optional; default is 0.5)
* `relative_score`: normalize probabilities by the largest probability (optional; default `false`)

Each of the labels should have a corresponding language model in
`lm_params`. The likelihood of the segment is calculated for all the
language models. If `relative_score` is false, the likelihoods are
normalized to probabilities that sum up to one over the labels, and
the probability of the expected label is returned as a score. If
`relative_score` is true, the probability values are first divided by
the largest probability (i.e. one of the labels will always get one as
the score).

For contents of the language model parameters in `lm_params`, see
[CrossEntropyFilter](CrossEntropyFilter).

See [train_ngram](train_ngram) for training the models. Note that the
format, perplexity calculation, and boundary marking options should
match the parameters used in model training; do not change them unless
you know what you are doing.

A possible use case for this filter is creating a custom language
identifier similar to Vatanen et al. (2010): Train a character-based
n-gram model for each of the languages from clean corpora, and use the
language codes as labels. Vatanen et al. (2010) recommend using
absolute discounting and maximum n-gram length 4 or 5 for the
models. Note that unknown tokens are ignored in the language model
likelihoods, so it is a good idea to train a small (e.g. unigram)
background model that includes data from all languages, and
interpolate the language-specific models with it using a small
interpolation coefficient. An example configuration is found at
[example_configs/qed_lm_langid.yaml](https://github.com/Helsinki-NLP/OpusFilter/blob/develop/example_configs/qed_lm_langid.yaml).
