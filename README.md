# OpusFilter

Tool for filtering parallel corpora.

## Installing

`pip install`

### Optional requirements

For using n-gram language model filters, you need to install VariKN
(https://github.com/vsiivola/variKN) and its Python bindings.

For using word alignment filters, you need to install elfomal
(https://github.com/robertostling/eflomal) and set environment
variable `EFLOMAL_PATH` to eflomal's root directory, which contains
the Python scripts `align.py` and `makepriors.py`.

## Filter configuration files

### Available functions

#### Downloading and selecting data

`opus_read`

`concatenate`

`subset`

#### Filtering and scoring

`filter`

`score`

#### Training models

`train_ngram`

`train_aligment`


### Available filters

`CrossEntropyFilter`

...
