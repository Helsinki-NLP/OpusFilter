# Tokenizer

Tokenize parallel texts.

Parameters:

* `tokenizer`: tokenizer type or a list of types for each input
* `languages`: a list of language codes for each input
* `options`: tokenizer options dictionary, or a list of tokenizer
  dictionaries for multiple tokenziers (optional)

Supported tokenizers:

* `moses`:
  * Uses the
    [opus-fast-mosestokenizer](https://github.com/Helsinki-NLP/opus-fast-mosestokenizer) package
    (a fork of [fast-mosestokenizer](https://github.com/mingruimingrui/fast-mosestokenizer)).
  * Available for most languages.
  * Options are passed to the `mosestokenizer.MosesTokenizer` class;
    see its documentation for the available options.
* `jieba`:
  * Uses the [jieba](https://github.com/fxsjy/jieba) package.
  * Only avaliable for Chinese (zh, zh_CN).
  * In order to keep track of original space characters, they are by
    default converted to "␣" before tokenization. The character can be
    changed with the `map_space_to` option, or the feature disabled by
    giving `null` or an empty string as the value.
  * Other options are passed to `jieba.cut` function; see its
    documentation for the avaliable options.
  * If you use `jieba`, please install OpusFilter with extras `[jieba]` or `[all]`.
* `mecab`:
  * Uses the [MeCab](https://github.com/SamuraiT/mecab-python3) package.
  * Only avaliable for Japanese (jp).
  * In order to keep track of original space characters, they are by
    default converted to "␣" before tokenization. The character can be
    changed with the `map_space_to` option, or the feature disabled by
    giving `null` or an empty string as the value.
  * By default, `unidic-lite` dictionary is installed and used. Other
    dictionaries can be used by providing appropriate option string in
    the `mecab_args` option.
  * If you use `mecab`, please install OpusFilter with extras `[mecab]` or `[all]`.

The list of language codes should match to the languages of the input
files given in the `preprocess` step. If more than on tokenizer is
provided, the length of the list should match the number of the
languages. If more than one tokenizer options are provided, the length
should again match the number of the languages.
