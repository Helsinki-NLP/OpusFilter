# RegExpSub

Run a sequence of arbitrary regular expression substitutions.

Parameters:

* `patterns`: a list of patterns to use by default
* `lang_patterns`: a dictionary or list of patterns to use for specific languages

Multiple substitutions are applied in the given order. The default
patterns are replaced with language-specific patterns when the
corresponding index (starting from 0) is found in the `lang_patterns`
dictionary. The `lang_patterns` argument may also be a list, if you
e.g. want to use separate patterns for all languages.

The substitution patterns are 4-tuples containing the regular
expression, replacement, count (0 = substitute all) and flags (list of
flag constants in the `re` library, e.g. `["I", "A"]`). The regular
expressions are first compiled with [`re.compile`](https://docs.python.org/3/library/re.html#re.compile),
and then the substitutions are applied with [`re.sub`](https://docs.python.org/3/library/re.html#re.sub).
