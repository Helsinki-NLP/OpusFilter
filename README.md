# Automatic configuration file generation

You can generate OpusFilter config files with the `opusfilter-autogen-cluster` script. The script takes a parallel corpus as an input and measures its quality to generate threshold parameters for filters. Currently, the used filters are AlphabetRatioFilter, CharacterScoreFilter, LanguageIDFilter, LengthRatioFilter, NonZeroNumeralsFilter and TerminalPunctuationFilter. This list will be expanded and made more flexible in the future.

## How does it work?

First, we remove duplicates and empty sentences from the input corpus. Next, we take a subset (100k sentencepairs by default) of the corpus and produce scores for each sentence pair in the subset with the previously mentioned filters. These scores are used as features for K-means clustering to classify the sentence pairs into clean and noisy pairs. The values of the noisy cluster center are used as the filter threshold parameters in the generated config file.

If you want to save the intermediate files, make sure to use the `--inter-dir` argument.

## usage:

```
opusfilter-autogen-cluster [-h] --files src_file trg_file --langs src_lang trg_lang
                                  [--scripts src_script trg_script] [--output-file OUTPUT_FILE]
                                  [--sample-size SAMPLE_SIZE] [--work-dir WORK_DIR]
                                  [--inter-dir INTER_DIR] [--graph] [--overwrite]

optional arguments:
  -h, --help            show this help message and exit
  --files src_file trg_file
                        Source and target file of a bitext
  --langs src_lang trg_lang
                        Language codes corresponding to the source and target files
  --scripts src_script trg_script
                        Alphabetic scripts corresponding to the source and target files. If omitted,
                        CharacterScoreFilter will not be used.
  --output-file OUTPUT_FILE
                        Generated config file (default=config.yaml)
  --sample-size SAMPLE_SIZE
                        Number of sentence pairs used for clustering (default=100000)
  --work-dir WORK_DIR   Location of the source and target files
  --inter-dir INTER_DIR
                        Save intermediate files in this directory. These files are: deduplicated files
                        > files with empty lines and lines longer than 150 words removed > sample files
                        for clustering > scores for the sample > labels from clustering
  --graph               Show a scatter plot of the clustering and histograms of feature data
                        distributions
  --overwrite           Overwrite existing config file and intermediate files
```

# OpusFilter

OpusFilter is a tool for filtering and combining parallel corpora.

Features:

* Corpus preprocessing pipelines configured with [YAML](https://yaml.org/)
* Simple downloading of parallel corpora from [OPUS](http://opus.nlpl.eu/) with [OpusTools](https://github.com/Helsinki-NLP/OpusTools)
* Implementations for many common text file operations on parallel files
* Memory-efficient processing of large files
* Implemented filters based e.g. on language identification, word aligment, n-gram language models, and multilingual sentence embeddings
* Extendable with your own filters written in Python

OpusFilter has been presented in [ACL 2020 system demonstrations](https://www.aclweb.org/anthology/2020.acl-demos.20).

## Installing

Install the latest release from PyPI:

* `pip install opusfilter` or `pip install opusfilter[all]` (include optional Python libraries)

Install from source:

* `pip install .` or `python setup.py install`

### Troubleshooting

OpusFilter should generally work fine on Python 3.6 to 3.10. In the case of troubles, try installing the exact versions in `requirements.txt`:

* `pip install -r requirements.txt`

## Documentation

The complete OpusFilter documentation is available from [helsinki-nlp.github.io/OpusFilter](https://helsinki-nlp.github.io/OpusFilter/).

You can also build the documents from the source:

* `pip install -r docs/requirements.txt` or  `pip install .[docs]`
* `sphinx-build docs docs-html`

## Changelog

A changelog is available in [docs/CHANGELOG.md](docs/CHANGELOG.md).

## Citing

If you use OpusFilter in your research, please cite our [ACL 2020 paper](https://www.aclweb.org/anthology/2020.acl-demos.20):

```bibtex
@inproceedings{aulamo-etal-2020-opusfilter,
    title = "{O}pus{F}ilter: A Configurable Parallel Corpus Filtering Toolbox",
    author = {Aulamo, Mikko and Virpioja, Sami and Tiedemann, J{\"o}rg},
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-demos.20",
    doi = "10.18653/v1/2020.acl-demos.20",
    pages = "150--156"
}
```

A full bibliography of papers cited in the documentation and code can be found from [docs/references.bib](docs/references.bib).

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).
