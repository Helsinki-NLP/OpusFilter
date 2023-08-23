# Sentence embedding filters

## SentenceEmbeddingFilter

Filter segments using sentence embeddings.

Parameters:

* `languages`: a list of language codes corresponding to input files
* `threshold`: filter out segments with similarity below the threshold (optional; default 0.5)
* `nn_model`: a nearest neighbor model for normalizing the similarities (optional; default `null`)
* `chunksize`: the number of segment pairs to process at the same time (optional: default 200)

The current implementation supports the multilingual LASER embeddings
as proposed by {cite:t}`artetxe-schwenk-2018-margin` and
{cite:t}`chaudhary-etal-2019-low`. Cosine similarity is used to
calculate the similarity of the embeddings.  If `nn_model` is
provided, the similarities are normalized by the average similarity to
K nearest neighbors in a reference corpus; see
[train_nearest_neighbors](train_nearest_neighbors) for training a
model. With normalized scores, threshold closer to 1.0 is likely more
suitable than the default 0.5.

Especially with the nearest neighbor normalization, this filter can be
slow to use. Using a small enough corpus for the nearest neighbors,
enabling GPU computation for PyTorch (used by `laserembeddings`), and
testing different values for `chunksize` may help.
