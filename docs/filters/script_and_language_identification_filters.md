# Script and language identification filters

## AlphabetRatioFilter

Filter segments based on what proportion of the characters are alphabetic characters.

Parameters:

* `threshold`: minimum proportion of alphabets in a segment (default 0.75)
* `exclude_whitespace`: whether to exclude whitespace characters from the ratio (default `false`)

Returned scores are proportions of alphabetic characters in the
segments (after removing whitespace if `exclude_whitespace` is
true). In filtering, all values have to be equal to or greater than
the minimum threshold.

In order to allow have different thresholds per language, the
threshold parameter can also be given as a list.

## CharacterScoreFilter

Filter segments based on what proportion of their alphabetic characters are in a given script. For a list of valid scripts, see e.g. [www.regular-expressions.info/unicode.html](https://www.regular-expressions.info/unicode.html).

Parameters:

* `scripts`: scripts for input segments
* `thresholds`: minimum proportion of characters in a script (default 1)

Returned scores are proportions of valid characters in the segments. In filtering, all values have to be equal to or greater than the minimum thresholds.

## LanguageIDFilter

Filter segments based on their language identification confidence scores.

Parameters:

* `languages`: expected languages (ISO639 language codes) for the segments
* `id_method`: language indentification method (`langid` for using the `langid` library, `cld2` for using the `cld2` library, or `fasttext` for using a `fasttext` model; the default is `langid`)
* `thresholds`: minimum identification confidence score for the segments (a single float or a list of floats per language)
* `fasttext_model_path`: path for a `fasttext` model (required only for the `fasttext` method; default `null`)
* `langid_languages`: limit detection to a list of possible languages (valid only for the `langid` method; default `null`)
* `cld2_options`: a dictionary of options for the `cld2` method (valid only for the `cld2` method; default `null`)

Returned scores are the language identification confidence scores from a given identification method for the segments. The scores range from 0 to 1. In filtering, all values have to be greater than the minimum thresholds. Negative threshold can be used to skip filtering for a language.

See [langid.py](https://github.com/saffsd/langid.py) and
[pycld2](https://github.com/aboSamoor/pycld2) for the method-specific
options. A pretrained `fasttext` model can be downloaded from
[fasttext.cc/docs/en/language-identification.html](https://fasttext.cc/docs/en/language-identification.html).
