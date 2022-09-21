# Custom preprocessors

Similarly to filters, You can import your own preprocessors by
defining the `module` key in the filter configuration entries.

The custom preprocessors should inherit the abstract base class
`PreprocessorABC` from the `opusfilter` package, and implement the
abstract method `process`. The `process` method is a generator that
takes an iterator over segments and yields preprocessed (modified)
segments of text. It also has an additional argument, `f_idx`, which
is the index of the current file being processed by the `preprocess`
step. This argument enables preprocessing parallel files in the same
step even if the preprocessing options (such as language code for a
tokenizer) varies.
