# OpusFilter

OpusFilter is a tool for filtering and combining parallel corpora. It
uses the OpusTools library (Aulamo et al., 2020) to download data from
the OPUS corpus collection (Tiedemann, 2012), but can be used with any
corpora in raw text format.

Features:

* Corpus preprocessing pipelines configured with [YAML](https://yaml.org/)
* Simple downloading of parallel corpora from [OPUS](http://opus.nlpl.eu/) with [OpusTools](https://github.com/Helsinki-NLP/OpusTools)
* Implementations for many common text file operations on parallel files
* Memory-efficient processing of large files
* Implemented filters based e.g. on language identification, word aligment, n-gram language models, and multilingual sentence embeddings
* Extendable with your own filters written in Python

OpusFilter has been presented in [ACL 2020 system demonstrations](https://www.aclweb.org/anthology/2020.acl-demos.20).

See the [documentation](https://github.com/) for more information and examples.

A changelog is available in [docs/CHANGELOG.md](docs/CHANGELOG.md).

## Citing and references

If you use OpusFilter in your research, please cite our [ACL 2020 paper](https://www.aclweb.org/anthology/2020.acl-demos.20):

```
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

A full bibliography of papers cited in the documentation and code can
be found from [docs/references.bib](docs/references.bib).

## How to contribute

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).