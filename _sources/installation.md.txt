# Installation

Install the latest release from PyPI:
* `pip install opusfilter`

Include optional Python libraries:
* `pip install opusfilter[all]`

Install from source:
* `pip install .` or
* `python setup.py install`

Note that all required libraries are not available to install via PyPI
on Windows OS. On Linux and MacOS, it should work directly for Python
versions from 3.8 to 3.12.

## Required libraries

* beautifulsoup4
* opus-fast-mosestokenizer
* graphviz
* py3langid
* matplotlib
* morfessor
* OpusTools
* pandas
* rapidfuzz
* ruamel.yaml
* regex
* requests
* sentence-splitter
* scikit-learn
* subword_nmt
* tqdm
* xxhash
* lingua-language-detector

See `setup.py` for possible version requirements.

## Optional libraries and tools

### FastText and PyCLD2 language identification

The language identification libraries currently supported out-of-the-box
are [py3langid](https://github.com/adbar/py3langid) and
[lingua](https://github.com/pemistahl/lingua-py). The support for for
[PyCLD2](https://github.com/aboSamoor/pycld2) and
[FastText models](https://fasttext.cc/docs/en/language-identification.html)
have been changed to optional due to the lack of support especially
for newer Python versions.

The PyCLD2 support can be installed automatically with pip by
including the extras `[pycld2]` or `[all]` (e.g.
`pip install opusfilter[pycld2]`).

The support for FastText models can be installed automatically with
pip by including the extras `[fasttext]` or `[all]` (e.g.
`pip install opusfilter[fasttext]`).

### Jieba and MeCab word segmentation

For Chinese tokenization (word segmentation), you can use the
[jieba](https://github.com/fxsjy/jieba) library. It can be installed
automatically with pip by including the extras `[jieba]` or `[all]`
(e.g. `pip install opusfilter[jieba]`).

For Japanese tokenization (word segmentation), you can use the
[MeCab](https://github.com/SamuraiT/mecab-python3) library. It can be installed
automatically with pip by including the extras `[mecab]` or `[all]`
(e.g. `pip install opusfilter[mecab]`).

### LASER sentence embeddings

For using sentence embeddings filters, you need to install
`laserembeddings` (https://github.com/yannvgn/laserembeddings). It can
be installed automatically with pip by including the extras `[laser]`
or `[all]` (e.g. `pip install opusfilter[laser]`). The package will also
require a number of additional libraries, including PyTorch, jieba,
and MeCab. Note that you need also to download the prebuild models
with `python -m laserembeddings download-models`.

### VariKN n-gram models

For using n-gram language model filters, you need to install the
Python wrapper for VariKN (https://github.com/vsiivola/variKN). It can
be installed automatically with pip by including the extras `[varikn]`
or `[all]` (e.g. `pip install opusfilter[varikn]`).

### Eflomal word alignment

For using word alignment filters, you need to install elfomal
(https://github.com/robertostling/eflomal). It can be installed
automatically with pip by including the extras `[eflomal]` or `[all]`
(e.g. `pip install opusfilter[eflomal]`). Note that you will need `Cython`
for the installation.
