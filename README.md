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

OpusFilter should generally work fine on Python 3.8 to 3.12. In the case of troubles, try installing the exact versions in `requirements.txt`:

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
