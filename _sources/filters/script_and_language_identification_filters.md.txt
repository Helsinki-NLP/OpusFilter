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

## LanguageIDFilter (deprecated)

*Note: The common LanguageIDFilter is now deprecated. Please see the method-specific language identification filters below.*

Filter segments based on their language identification confidence scores.

Parameters:

* `languages`: expected languages (ISO639 language codes) for the segments
* `id_method`: language indentification method (`langid`, `lingua`, `cld2`, `fasttext`, `heliport`; default `langid`)
* `thresholds`: minimum identification confidence score for the segments (a single float or a list of floats per language)
* `fasttext_model_path`: path for a `fasttext` model (required only for the `fasttext` method; default `null`)
* `langid_languages`: limit detection to a list of ISO 639-1 codes for possible languages (valid only for the `langid` and `lingua` methods; default `null`)
* `cld2_options`: a dictionary of options for the `cld2` method (valid only for the `cld2` method; default `null`)
* `lingua_mode`: a string specifying whether to use lingua's `high` or `low` accuracy mode

Returned scores are the language identification confidence scores from a given identification method for the segments. The scores range from 0 to 1. In filtering, all values have to be greater than the minimum thresholds. A negative threshold can be used to skip filtering for a language.

Currently the following identification methods are supported:

* `langid` (default) :cite:`lui-baldwin-2012-langid`
  * See https://github.com/adbar/py3langid
* `lingua`
  * See https://github.com/pemistahl/lingua-py
* `cld2`
  * See https://github.com/CLD2Owners/cld2
  * Requires [installing optional libraries](../installation.md).
* `fasttext` :cite:`joulin-etal-2016-fasttext` and :cite:`joulin-etal-2017-bag`
  * A pretrained model can be downloaded from [fasttext.cc/docs/en/language-identification.html](https://fasttext.cc/docs/en/language-identification.html).
  * Requires [installing optional libraries](../installation.md).

## LangidFilter

Filter segments based on the langid method :cite:`lui-baldwin-2012-langid`; see
https://github.com/adbar/py3langid

Parameters:

* `languages`: expected languages (ISO639 language codes) for the segments
* `thresholds`: minimum identification confidence score for the segments (a single float or a list of floats per language)
* `langid_languages`: limit detection to a list of ISO 639-1 codes for possible languages

Returned scores are the language identification confidence scores for
the segments. The scores range from 0 to 1. In filtering, all values
have to be greater than the minimum thresholds. A negative threshold
can be used to skip filtering for a language.

## Cld2Filter

Filter segments based on the CLD2 method; see https://github.com/CLD2Owners/cld2

Requires [installing optional libraries](../installation.md).

Parameters:

* `languages`: expected languages (ISO639 language codes) for the segments
* `thresholds`: minimum identification confidence score for the segments (a single float or a list of floats per language)
* `options`: a dictionary of options for the method (see https://github.com/CLD2Owners/cld2; default `null`)

Returned scores are the language identification confidence scores from
a given identification method for the segments. The scores range from
0 to 1. In filtering, all values have to be greater than the minimum
thresholds. A negative threshold can be used to skip filtering for a
language.

## FastTextFilter

Filter segments based on the Fasttext language identification models
(:cite:`joulin-etal-2016-fasttext`, :cite:`joulin-etal-2017-bag`), see
https://github.com/CLD2Owners/cld2.

Requires [installing optional libraries](../installation.md).

Parameters:

* `languages`: expected languages (ISO639 language codes) for the segments
* `thresholds`: minimum identification confidence score for the segments (a single float or a list of floats per language)
* `model_path`: path for a `fasttext` model

Returned scores are the language identification confidence scores from
a given identification method for the segments. The scores range from
0 to 1. In filtering, all values have to be greater than the minimum
thresholds. A negative threshold can be used to skip filtering for a
language.

## LinguaFilter

Filter segments based on the Lingua method; see https://github.com/pemistahl/lingua-py

Parameters:

* `languages`: expected languages (ISO639 language codes) for the segments
* `thresholds`: minimum identification confidence score for the segments (a single float or a list of floats per language)
* `langid_languages`: limit detection to a list of ISO 639-1 codes for possible languages (default `null`)
* `lingua_mode`: a string specifying whether to use lingua's `high` or `low` accuracy mode (default `low`)

Returned scores are the language identification confidence scores from
a given identification method for the segments. The scores range from
0 to 1. In filtering, all values have to be greater than the minimum
thresholds. A negative threshold can be used to skip filtering for a
language.

## HeliportSimpleFilter

Filter segments based on the HeLI-OTS method :cite:`jauhiainen-etal-2022-heli`; see
https://github.com/ZJaume/heliport

Requires [installing optional libraries](../installation.md).

Parameters:

* `languages`: expected languages (ISO639 language codes) for the segments
* `thresholds`: minimum identification score for the segments (a single float or a list of floats per language)

This version uses three distinct score values. Score 1.0 is used if
the method predicts the correct language. If the low-confidence label
`und` is returned, the language is considered correct with score
0.5. Finally, if a wrong language is predicted, the score will
0.0. A negative threshold can be used to skip filtering for a language.

## HeliportConfidenceFilter

Filter segments based on the HeLI-OTS method :cite:`jauhiainen-etal-2022-heli`; see
https://github.com/ZJaume/heliport

Requires [installing optional libraries](../installation.md).

Parameters:

* `languages`: expected languages (ISO639 language codes) for the segments
* `thresholds`: minimum identification score for the segments (a single float or a list of floats per language)

This version uses the confidence value returned by the method. It is
the difference between raw scores of the the best (lowest) and second
best language. The maximum value is 7.0, which is a penalty value for
the languages that do not match. The minimum value 0 is returned if
the best language does not match. The low-confidence label `und` is
considered as a match. A negative threshold can be used to skip
filtering for a language.

## HeliportRawScoreFilter

Filter segments based on the HeLI-OTS method :cite:`jauhiainen-etal-2022-heli`; see
https://github.com/ZJaume/heliport

Requires [installing optional libraries](../installation.md).

Parameters:

* `languages`: expected languages (ISO639 language codes) for the segments
* `thresholds`: minimum identification score for the segments (a single float or a list of floats per language)
* `topk`: the number of top languages to consider (default 50)

This version uses the raw scores returned by the method. A lower score
is better. If the correct language is among the top-k languages, the
score if that language is returned. Otherwise, the penalty value 7 is
returned. A threshold over the penalty value can be used to skip
filtering for a language.

## HeliportProbabilityFilter

Filter segments based on the HeLI-OTS method :cite:`jauhiainen-etal-2022-heli`; see
https://github.com/ZJaume/heliport

Requires [installing optional libraries](../installation.md).

* `languages`: expected languages (ISO639 language codes) for the segments
* `thresholds`: minimum identification score for the segments (a single float or a list of floats per language)
* `topk`: the number of top languages to consider (default 50)

This version transforms the raw scores of of the top-k languages into
probability-like values. If the correct language is not in the top-k
list, zero is returned. A negative threshold can be used to skip
filtering for a language.
