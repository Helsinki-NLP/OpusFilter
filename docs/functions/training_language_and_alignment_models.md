# Training language and alignment models

## train_ngram

Train a character-based varigram language model with VariKN
{cite:p}`siivola-etal-2007-growing`. Can be used for
[`CrossEntropyFilter`](CrossEntropyFilter) and
[`CrossEntropyDifferenceFilter`](CrossEntropyDifferenceFilter).

Parameters:

* `data`: input file name for training data
* `model`: output file name for the model
* `parameters`: training options for VariKN and tokenization
   * `optdata`: filename for optimization data (optional; default empty string `""` = use leave-one-out estimation instead)
   * `norder`: limit model order (optional; default 0 = no limit)
   * `dscale`: model size scale factor (optional; smaller value gives a larger model; default 0.001)
   * `dscale2`: model size scaling during pruning step (optional; default 0 = no pruning)
   * `arpa`: output ARPA instead of binary LM (optional; default `true`)
   * `use_3nzer`: use 3 discounts per order instead of one (optional; default `false`)
   * `absolute`: use absolute discounting instead of Kneser-Ney smoothing (optional; default `false`)
   * `cutoffs`: use the specified cutoffs (optional; default `"0 0 1"`). The last value is used for all higher order n-grams.
   * `segmentation`: subword segmentation options (optional; default `{}`)
   * `mb`: word-internal boundary marking (optional; default `""`)
   * `wb`: word boundary tag (optional; default `"<w>"`)

Apart from the scale, cutoff, and order parameters the size of the
model depends on the size of the training data. Typically you want to
at least change the `dscale` value to get a model of a reasonable
size. If unsure, start with high values, look at the number of the
n-grams in the output file, and divide by 10 if it looks too small.
The `dscale2` option is useful mostly if you want to optimize the
balance between the model size and accuracy at the cost of longer
training time; a suitable rule of thumb is double the value of
`dscale`.

The `segmentation` parameter is a dictionary that should contain at
least the key `type`, which defines the subword segmentation type.
The default is character segmentation (`type: char`). Other options
are no segmentation (`type: none`), and BPE (`type: bpe`) or Morfessor
(`type: morfessor`) segmentations. For the latter two, a file for a
trained segmentation model needs to be defined using the key `model`.
Additional parameters in the dictionary are passed as options for the
specified model; see [`BPESegmentation`](BPESegmentation) and
[`MorfessorSegmentation`](MorfessorSegmentation) for those. The BPE
and Morfessor models can be trained using the [`train_bpe`](train_bpe)
and [`train_morfessor`](train_morfessor) commands.

The default boundary settings (a separate word boundary tag) are
suitable for character-based models. For other subword models, you
may consider using the word-internal boundary marking (`mb`)
instead. Either prefix or postfix string can be used: Prefix strings
start with `^` and postfix strings end with `$`. For example,
`mb: "^#"` means that a token starting with `#` is not preceeded by a
word break (e.g. `sub #word segment #tation`). The postfix marking
used by `subword-nmt` (e.g. `sub@@ word segment@@ ation`) can be set
by `mb: "@@$"`.

See [VariKN](https://github.com/vsiivola/variKN) documentation for
details.

## train_aligment

Train word alignment priors for eflomal {cite:p}`ostling-tiedemann-2016-efficient`.
Can be used in [`WordAlignFilter`](WordAlignFilter).

Parameters:

* `src_data`: input file for the source language
* `tgt_data`: input file for the target language
* `parameters`: training options for the aligment and tokenization
   * `src_tokenizer`: tokenizer for source language (optional; default `null`)
   * `tgt_tokenizer`: tokenizer for target language (optional; default `null`)
   * `model`: eflomal model type (optional; default 3)
* `output`: output file name for the priors
* `scores`: file to write alignment scores from the training data (optional; default `null`)

See [`WordAlignFilter`](WordAlignFilter) for details of the training
parameters.

## train_nearest_neighbors

Train unsupervised model to search for nearest neighbors of segments using sentence
embeddings. Can be used in [`SentenceEmbeddingFilter`](SentenceEmbeddingFilter).

Parameters:

* `inputs`: a list of input files
* `languages`: a list of language codes corresponding to the input files
* `n_neighbors`: the default number neightbors to return from query (optional; default 4)
* `algorithm`: algorithm used to compute the nearest neighbors (optional; default `brute`)
* `metric`: distance or similarity metric used by the object (optional; default `cosine`) 
* `output`: output file name for the model

This is a wrapper for scikit-learn's `NearestNeighbors` class; see more information in it's
[documentation](https://scikit-learn.org/stable/modules/neighbors.html#unsupervised-neighbors).
Note that the cosine similarity is required for proper use in `SentenceEmbeddingFilter`,
and only the brute force algorithm works with cosine similarities. The saved model can be
very large, so use large input corpora with caution.

## train_bpe

Train subword segmentation model with BPE {cite:p}`sennrich-etal-2016-neural`.

Parameters:

* `input`: input file name for training data
* `model`: output file name for the model
* `symbols`: create this many new symbols (each representing a character n-gram) (optional; default 10000)
* `min_frequency`: stop if no symbol pair has frequency equal or above the threshold (optional; default 2)
* `num_workers`: number of processors to process texts; if -1, set `multiprocessing.cpu_count()` (optional; default 1)

See [subword-nmt documentation](https://github.com/rsennrich/subword-nmt) for details.
The trained model can be used by the [`BPESegmentation`](BPESegmentation) preprocessor.

## train_morfessor

Train subword segmentation model with Morfessor 2.0 {cite:p}`virpioja-etal-2013-morfessor`.

Parameters:

* `input`: input file name for training data
* `model`: output file name for the model
* `corpusweight`: corpus weight parameter (optional; default 1.0)
* `min_frequency`: frequency threshold for words to include in training (optional; default 1)
* `dampening`: frequency dampening for training data: `none` = tokens, `log` = logarithmic dampening, or `ones` = types (optional; default `"log"`)
* `seed`: seed for random number generator used in training (optional; default `null`)
* `use_skips`: use random skips for frequently seen compounds to speed up training (optional; default `true`)
* `forcesplit_list`: force segmentations on the characters in the given list (optional; default `null`)
* `nosplit_re`: if the regular expression matches the two surrounding characters, do not allow splitting (optional; default `null`)

See [Morfessor 2.0 documentation](https://morfessor.readthedocs.io/en/latest/) for details.
The trained model can be used by the [`MorfessorSegmentation`](MorfessorSegmentation) preprocessor.
