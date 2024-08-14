# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.2.0] - 2024-08-14

### Changed

- make `pycld2` and `fasttext` libraries optional
- replace `langid.py` library with `py3langid`
- update github workflows and include Python 3.12 tests

### Fixed

- `OpusRead` interface using `moses` format (requires `opustools >= 1.6.2`)

## [3.1.0] - 2024-06-05

### Added

- support `lingua` based for language detection (https://github.com/Helsinki-NLP/OpusFilter/pull/65)

### Removed

- Python 3.7 support

### Fixed

- fix score method in `SentenceEmbeddingFilter` (https://github.com/Helsinki-NLP/OpusFilter/pull/71)
- fix filter and filterfalse methods in `SentenceEmbeddingFilter`

## [3.0.0] - 2023-10-11

### Added

- `opusfilter-autogen` script for automatic filter config generation
- `score_direction`, `accept_threshold`, and `reject_threshold` properties for filters

### Changed

- refactor code and move auxiliary methods to opusfilter.util
- update varikn installation instructions (installable from PyPI)
- update github workflows and include Python 3.11 tests
- update library version requirements to support Python 3.11
- use xxhash instead of pyhash for hash functions
- use opus-fast-mosestokenizer instead of fast-mosestokenizer
- install eflomal from PyPI and use the new interface in WordAlignFilter

### Removed

- Python 3.6 support

### Fixed

- catch NotImplementedError from beautifulsoup 4.11.2
- catch ParserRejectedMarkup from beautifulsoup 4.12.0

## [2.6.0] - 2022-11-30

### Added

- add `slice` missing from the enabled steps

### Changed

- improve documentation
- import slow libraries only when needed
- use chunks for the filter method of `SentenceEmbeddingFilter`
- change `RepetitionFilter` to use single score for consistency with the threshold

### Fixed

- allow float thresholds for `AverageWordLengthFilter`
- remove unnecessary code from `RegExpSub`
- add `setuptools` version requirement

## [2.5.1] - 2022-09-28

### Fixed

- add missing document file

## [2.5.0] - 2022-09-28

### Added

- `map_space_to` option for Jieba and MeCab tokenizers to preserve
  existing space characters in input
- parallel processing options for filter, score, and preprocess steps

### Changed

- re-organize documentation and support building it with sphinx

### Fixed

- catch TypeError exceptions from BeautifulSoup in HtmlTagFilter

## [2.4.0] - 2022-04-05

### Added

- an option to write filter scores to a file with `opusfilter-test`
- new filters: `AlphabetRatioFilter`, `RegExpFilter`, `SimilarityFilter`, `SentenceEmbeddingFilter`
- support for Japanese word segmentation using `MeCab` as a tokenizer
- preprocessing methods for subword segmentation (`BPESegmentation`, `MorfessorSegmentation`)
- subword segmentation support for the n-gram language models and language model filters

### Changed

- allow per-language parameters for LengthFilter, LengthRatioFilter, LongWordFilter, and AverageWordLengthFilter
- fix documentation for `train_aligment` parameters

## [2.3.1] - 2022-01-28

### Fixed

- fix bug in classifier training without development set

## [2.3.0] - 2022-01-18

### Added

- new OpusFilterRuntimeError exception for having e.g. empty training data
- option to save scores from the training data when creating word aligment priors
- RepetitionFilter for filtering segments with repeated substrings
- new preprocessor for sentence splitting monolingual data
- method-specific options for LanguageIDFilter
- chunksize option to the common section
- LMClassifierFilter for classification based on n-gram language models

### Changed

- add `workdir` attribute to the `FilterABC` base class and change
  that the filters should use it for any file parameters
- increase default chunksize in FilterPipeline from 10000 to 100000
- refactor and clean up code

## [2.2.0] - 2021-11-23

### Added

- support for Chinese word segmentation using `jieba` as a tokenizer (https://github.com/Helsinki-NLP/OpusFilter/pull/27)

## [2.1.2] - 2021-11-11

### Fixed

- fix wrong keyword argument name in opusfilter-duplicates

## [2.1.1] - 2021-10-19

### Changed

- move "How to contribute" to docs/CONTRIBUTING.md

### Fixed

- fix setuptools requirement (https://github.com/Helsinki-NLP/OpusFilter/issues/21)
- fix version requirement for pandas (>=1.0.0)

## [2.1.0] - 2021-08-31

### Changed

- replace PyYAML with ruamel.yaml

### Added

- support for variables in the YAML configuration (https://github.com/Helsinki-NLP/OpusFilter/pull/13)
- support to `fasttext` based for language detection (https://github.com/Helsinki-NLP/OpusFilter/pull/20)
- `suppress_prompts` parameter for `opus_read` (https://github.com/Helsinki-NLP/OpusFilter/pull/19)
- `download` and `write` steps
- "How to contribute" section to README.md
- changelog
- bibliography and improved references

## [2.0.0] - 2021-06-01

### Changed

- extend to n-lingual parallel data instead of just bilingual data
- switch tokenizer to `fast-mosestokenizer`

### Added

- new commands: `opusfilter-diagram`, `opusfilter-duplicates`, `opusfilter-test`
- new filters: `LongestCommonSubstringFilter`, `AverageWordLengthFilter`
- new steps: `preprocess`
- set "latest" as the default corpus release for `opus_read` (https://github.com/Helsinki-NLP/OpusFilter/pull/5)
- overlap option for `remove_duplicates`
- lower threshold option for `CrossEntropyFilter`
- github CI workflow for flake8 and unittests

### Fixed

- behaviour of simple filters on empty segments

## [1.0.1] - 2020-05-25

### Added

- improved logging, documentation, and project files

### Fixed

- prevent `UnboundLocalError` for empty output after filter

## [1.0.0] - 2020-04-10

First tagged version.


[Unreleased]: https://github.com/Helsinki-NLP/OpusFilter/compare/3.2.0...develop
[3.2.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/3.1.0...3.2.0
[3.1.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/3.0.0...3.1.0
[3.0.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.6.0...3.0.0
[2.6.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.5.1...2.6.0
[2.5.1]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.5.0...2.5.1
[2.5.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.4.0...2.5.0
[2.4.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.3.1...2.4.0
[2.3.1]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.3.0...2.3.1
[2.3.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.2.0...2.3.0
[2.2.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.1.2...2.2.0
[2.1.2]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.1.1...2.1.2
[2.1.1]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.1.0...2.1.1
[2.1.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.0.0...2.1.0
[2.0.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/1.0.1...2.0.0
[1.0.1]: https://github.com/Helsinki-NLP/OpusFilter/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/Helsinki-NLP/OpusFilter/tree/1.0.0
