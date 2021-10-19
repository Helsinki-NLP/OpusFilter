# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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


[Unreleased]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.1.1...develop
[2.1.1]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.1.0...2.1.1
[2.1.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/2.0.0...2.1.0
[2.0.0]: https://github.com/Helsinki-NLP/OpusFilter/compare/1.0.1...2.0.0
[1.0.1]: https://github.com/Helsinki-NLP/OpusFilter/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/Helsinki-NLP/OpusFilter/tree/1.0.0
