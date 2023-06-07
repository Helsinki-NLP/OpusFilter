.. Opusfilter documentation master file, created by
   sphinx-quickstart on Fri Sep 16 21:26:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpusFilter
==========

Welcome to OpusFilter's documentation!

OpusFilter is a tool for filtering and combining parallel corpora. It
uses the OpusTools library :cite:p:`aulamo-etal-2020-opustools` to
download data from the OPUS corpus collection
:cite:p:`tiedemann-2012-parallel`, but can be used with any corpora in
raw text format.

Features:

- Corpus preprocessing pipelines configured with YAML
- Simple downloading of parallel corpora from OPUS with OpusTools
- Implementations for many common text file operations on parallel files
- Memory-efficient processing of large files
- Implemented filters based e.g. on language identification, word aligment, n-gram language models, and multilingual sentence embeddings
- Extendable with your own filters written in Python

OpusFilter has been presented in `ACL 2020 system demonstrations <https://www.aclweb.org/anthology/2020.acl-demos.20>`_.

.. toctree::
   :caption: Get started
   :maxdepth: 1

   installation.md
   usage.md
   automatic_configuration.md
   command_line_tools.md

.. toctree::
   :caption: Available functions
   :name: functions
   :maxdepth: 1

   functions/downloading_and_selecting_data.md
   functions/preprocessing_text.md
   functions/filtering_and_scoring.md
   functions/using_score_files.md
   functions/training_language_and_alignment_models.md
   functions/training_and_using_classifiers.md

.. toctree::
   :caption: Available filters
   :name: filters
   :maxdepth: 1
   :glob:

   filters/length_filters.md
   filters/script_and_language_identification_filters.md
   filters/special_character_and_similarity_filters.md
   filters/language_model_filters.md
   filters/alignment_model_filters.md
   filters/sentence_embedding_filters.md
   filters/custom_filters.md

.. toctree::
   :caption: Available preprocessors
   :name: preprocessors
   :maxdepth: 1

   preprocessors/tokenizer.md
   preprocessors/detokenizer.md
   preprocessors/whitespaceNormalizer.md
   preprocessors/reg_exp_sub.md
   preprocessors/monolingual_sentence_splitter.md
   preprocessors/bpe_segmentation.md
   preprocessors/morfessor_segmentation.md
   preprocessors/custom_preprocessors.md

.. toctree::
   :caption: Other information
   :maxdepth: 1
   
   references.rst
   CONTRIBUTING.md
   CHANGELOG.md
