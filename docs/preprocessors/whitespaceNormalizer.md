# WhitespaceNormalizer

Normalizes whitespace characters.

The whitespace normalization consists of two steps: First, any
sequence of one or more unicode whitespace characters (as matched by
`\s` in the `re` standard library) are replaced by a single space
character. Second, any leading and trailing whitespace is removed.
