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
versions from 3.8 to 3.11.

## Required libraries

* beautifulsoup4
* opus-fast-mosestokenizer
* fasttext
* graphviz
* langid
* matplotlib
* morfessor
* OpusTools
* pandas
* pycld2
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

### Jieba and MeCab word segmentation

For Chinese tokenization (word segmentation), you can use the
[jieba](https://github.com/fxsjy/jieba) library. It can be installed
automatically with pip by including the extras `[jieba]` or `[all]`
(e.g. `pip install opusfilter[all]`).

For Japanese tokenization (word segmentation), you can use the
[MeCab](https://github.com/SamuraiT/mecab-python3) library. It can be installed
automatically with pip by including the extras `[mecab]` or `[all]`
(e.g. `pip install opusfilter[all]`).

### LASER sentence embeddings

For using sentence embeddings filters, you need to install
`laserembeddings` (https://github.com/yannvgn/laserembeddings). It can
be installed automatically with pip by including the extras `[laser]`
or `[all]` (e.g. `pip install opusfilter[all]`). The package will also
require a number of additional libraries, including PyTorch, jieba,
and MeCab. Note that you need also to download the prebuild models
with `python -m laserembeddings download-models`.

### VariKN n-gram models

For using n-gram language model filters, you need to install the
Python wrapper for VariKN (https://github.com/vsiivola/variKN). It can
be installed automatically with pip by including the extras `[varikn]`
or `[all]` (e.g. `pip install opusfilter[all]`).

### Eflomal word alignment

For using word alignment filters, you need to install elfomal
(https://github.com/robertostling/eflomal). It can be installed
automatically with pip by including the extras `[eflomal]` or `[all]`
(e.g. `pip install opusfilter[all]`). Note that you will need `Cython`
for the installation.
