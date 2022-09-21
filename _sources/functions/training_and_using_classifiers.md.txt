# Training and using classifiers

## train_classifier

Train an `sklearn` classifier to produce a cleanness score for sentence pairs.

Parameters:

* `training_scores`: a file containing filter scores for training in JSON lines format produced with `score` function.
* `criterion`: criterion to be used in classifier optimization (valid options are `CE`, `ROC_AUC`, `SSE`, `AIC` and `BIC`)
* `dev_scores`: a file containing filter scores for training in JSON lines format produced with `score` function with and added item `label` added to each entry. `label` has value 1 for clean pairs and 0 for noisy pairs (optional; `dev_scores` is only used when the `criterion` is `ROC_AUC`)
* `model_type`: classifier model type selected from `sklearn` classifiers (default `LogisticRegression`)
* `model_parameters`: parameters for the `sklearn` classifier
* `model`: output model file
* `features`: the features given to the classifier to be trained on, defined as a list of filter names
    * `ExampleFilter`:
        * `clean-direction`: the direction that indicates higher cleanness (valid options are `high` and `low`)
        * `quantiles`: a dictionary the items of which (`min`, `max` and `initial`) specify the minimum, maximum and inital quantile value that are used in classifier optimization to select negative and positive training examples (default `{'min': 0, 'max': 1, 'initial': 0.1}`)

The classifier is optimized by training multiple classifier model with the training data divided differently into positive and negative examples based on the quantile boundaries specified in each feature. The model that achieves the highest criterion score is then saved in the output file.

## classify

Use a classifier model trained with `train_classifier` to assign a cleanness score or label to sentence pairs that have been scored with `score`.

Parameters:

* `model`: classifier model trained with `train_classifier`
* `scores`: scores of the sentence pairs to be classifed in JSON lines format produced with the `score` function
* `output_probabilities`: file to write the cleanness scores to, 1 is cleanest and 0 is noisiest (optional)
* `output_labels`: file to write the cleanness labels to, 1 is a clean and 0 is a noisy pair (optional)

The probabilities and labels are written to the output files line by line, corresponding to the scores on each line in `scores`.
