# Custom filters

You can also import your own filters by defining the `module` key in
the filter configuration entries.

The custom filters should inherit the abstract base class `FilterABC`
from the `opusfilter` package. They should implement two abstract
methods, `score` and `accept`, and one abstract property,
`score_direction`. Additionally, for filters with adjustable
thresholds, defining `accept_threshold` and `reject_threshold`
properties is recommended.

The `score` method is a generator that takes an iterator over tuples
of parallel sentences, and yields a score object for each pair. The
score may either be a single number, or if multiple score values need
to be yielded, a dictionary that has the numbers as values.

The `accept` method takes a single output yielded by the `score`
method, and returns whether the sentence pair should be accepted based
on the score.

The `score_direction` should be one of the following constants defined
in the `opusfilter` module depending on the output of the `score()`
method:

* `CLEAN_LOW`: scores below a threshold parameter indicate clean data
* `CLEAN_HIGH`: scores above a threshold parameter indicate clean data
* `CLEAN_BETWEEN`: scores between minimum and maximum thresholds
  indicate clean data
* `CLEAN_TRUE`: score value `True` indicates clean data
* `CLEAN_FALSE`: score value `False` indicates clean data

If the filter requires any parameters (e.g. score thresholds for the
`accept` method), the class should implement also the `__init__`
method.  Arbitrary keyword arguments should be accepted (with
`**kwargs`), and the `__init__` method of the base class (`FilterABC`)
should be called with the remaining keyword arguments. The keyword
argument `name` is reserved for giving names to the filters and
`workdir` for a location for non-temprary files.

For compability with the included [automatic configuration generation
tools](../automatic_configuration.md), also the following should be
considered:

* If there is a threshold value used by `accept`, the argument should
  be named as `threshold` (a single global threshold) or `thresholds`
  (multiple thresholds, e.g. one per language). The `accept_threshold`
  and `reject_threshold` properties should have threshold values that
  force all inputs to be accepted or rejected, respectively.  That is,
  a sensible threshold value will always be between `accept_threshold`
  and `reject_threshold`.
* If there are lower and upper thresholds used by `accept`
  (i.e. `score_direction` is `CLEAN_BETWEEN`), the respective
  arguments should be named as `min_threshold` and `max_threshold` or
  `min_length` and `max_length`. The `accept_threshold` and
  `reject_threshold` properties should have tuples of two threshold
  values (for lower and upper thresholds) that force all inputs to be
  accepted or rejected, respectively.

Based on the `score` and `accept` methods, the abstract class
`FilterABC` implements the following three generators that take
iterator over segment pairs as input:

* `decisions` yields results of the `accept` method
* `filter` yields only accepted segments
* `filterfalse` yields only rejected segments

These should not be redefined except for a good reason.

The example below shows code for simple filter that calculates the
proportion of uppercase letters in the sentences, and accepts the pair
only if all sentences have less than 50% (or given threshold) of
uppercase characters:

```python
import opusfilter

class UppercaseFilter(opusfilter.FilterABC):

	score_direction = opusfilter.CLEAN_LOW
	accept_threshold = 1 + 10**-6
	reject_threshold = 0

    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def uppercase_ratio(self, sentence):
        length = len(sentence)
        if length > 0:
            return sum(1 for char in sent if char.isupper()) / length
        return 0

    def score(self, pairs):
        for pair in pairs:
            yield [self.uppercase_ratio(sentence) for sentence in pair]

    def accept(self, score):
        return all(ratio < self.threshold for ratio in score)
```

Assuming that the above code is in a module named `customfilter` in
the Python evironment (e.g. save the code as `customfilter.py` and add
the directory that contains it to `PYTHONPATH` environment variable),
it can be selected in the filter configurations as follows:

```yaml
steps:

  ...

  - type: filter
    parameters:

      ...

      filters:

        - UppercaseFilter:
            threshold: 0.5
          module: customfilter
```

If a filter requires external resources files (e.g. for model
parameters), or stores non-temporary files itself, they should be
located in the path defined the attribute `workdir`. The
implementation of the filter should join `workdir` with relative file
paths using `os.path.join()`.
