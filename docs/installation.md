# Installation
Install the latest release from PyPI:
* `pip install opusfilter`

Include optional Python libraries:
* `pip install opusfilter[all]`

Install from source:
* `pip install .` or
* `python setup.py install`

Note that all required libraries are not available to install via PyPI
on Windows OS.  On Linux, it should work directly for Python versions
from 3.6 to 3.8, but with Python 3.9 the `fast-mosestokenizer` library
currently requires a manual install.

## Required libraries

* beautifulsoup4
* graphviz
* langid
* fast-mosestokenizer
* fasttext
* matplotlib
* morfessor
* OpusTools
* pandas
* pycld2
* pyhash
* rapidfuzz
* ruamel.yaml
* regex
* requests
* sentence-splitter
* scikit-learn
* subword_nmt
* tqdm

See `setup.py` for possible version requirements.

## Optional libraries and tools

For Chinese tokenization (word segmentation), you can use the
[jieba](https://github.com/fxsjy/jieba) library. It can be installed
automatically with pip by including the extras `[jieba]` or `[all]`
(e.g. `pip install opusfilter[all]`).

For Japanese tokenization (word segmentation), you can use the
[MeCab](https://github.com/SamuraiT/mecab-python3) library. It can be installed
automatically with pip by including the extras `[mecab]` or `[all]`
(e.g. `pip install opusfilter[all]`).

For using sentence embeddings filters, you need to install
`laserembeddings` (https://github.com/yannvgn/laserembeddings). It can
be installed automatically with pip by including the extras `[laser]`
or `[all]` (e.g. `pip install opusfilter[all]`). The package will also
require a number of additional libraries, including PyTorch, jieba,
and MeCab.

For using n-gram language model filters, you need to install VariKN
(https://github.com/vsiivola/variKN) and its Python wrapper. Include
the library files compiled to `build/lib/python` to your Python
library path (e.g. by setting the `PYTHONPATH` environment variable).

For using word alignment filters, you need to install elfomal
(https://github.com/robertostling/eflomal) and set environment
variable `EFLOMAL_PATH` to eflomal's root directory, which contains
the Python scripts `align.py` and `makepriors.py`. Note that you
will need `Cython` to install the Python interface to `eflomal`.