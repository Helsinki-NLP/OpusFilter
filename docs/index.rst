.. Opusfilter documentation master file, created by
   sphinx-quickstart on Fri Sep 16 21:26:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpusFilter
==========

Welcome to Opusfilter's documentation!

OpusFilter is a tool for filtering and combining parallel corpora. It uses the OpusTools library (Aulamo et al., 2020) to download data from the OPUS corpus collection (Tiedemann, 2012), but can be used with any corpora in raw text format.

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
   quickstart.md
   

.. toctree::
   :caption: Available functions
   :maxdepth: 1

   downloading_and_selecting_data.md
   preprocessing_text.md
   filtering_and_scoring.md
   using_score_files.md
   training_language_and_alignment_models.md
   training_and_using_classifiers.md

.. toctree::
   :caption: Avaliable Filters
   :maxdepth: 1
   :glob:

   length_filters.md
   script_and_language_identification_filters.md
   special_character_and_similarity_filters.md
   language_model_filters.md
   alignment_model_filters.md
   sentence_embedding_filters.md
   custom_filters.md

.. toctree::
   :caption: Avaliable Preprocessors
   :maxdepth: 1

   tokenizer.md
   detokenizer.md
   whitespaceNormalizer.md
   reg_exp_sub.md
   monolingual_sentence_splitter.md
   bpe_segmentation.md
   morfessor_segmentation.md
   custom_preprocessors.md


.. toctree::
   :caption: Other information
   :maxdepth: 1
   
   other_tools.md
   CONTRIBUTING.md
   CHANGELOG.md