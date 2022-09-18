# Custom filters

You can also import your own filters by defining the `module` key in
the filter configuration entries.

The custom filters should inherit the abstract base class `FilterABC`
from the `opusfilter` package. They should implement two abstract
methods: `score` and `accept`.

The `score` method is a generator that takes an iterator over tuples
of parallel sentences, and yields a score object for each pair. The
score may either be a single number, or if multiple score values need
to be yielded, a dictionary that has the numbers as values.

The `accept` method takes a single output yielded by the `score`
method, and returns whether the sentence pair should be accepted based
on the score.

If the filter requires any parameters (e.g. score thresholds for the
`accept` method), the class should implement also the `__init__`
method.  Arbitrary keyword arguments should be accepted (with
`**kwargs`), and the `__init__` method of the base class (`FilterABC`)
should be called with the remaining keyword arguments. The keyword
argument `name` is reserved for giving names to the filters and
`workdir` for a location for non-temprary files.

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