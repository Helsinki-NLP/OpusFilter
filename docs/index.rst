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

   avaliable_functions/downloading_and_selecting_data.md
   avaliable_functions/preprocessing_text.md
   avaliable_functions/filtering_and_scoring.md
   avaliable_functions/using_score_files.md
   avaliable_functions/training_language_and_alignment_models.md
   avaliable_functions/training_and_using_classifiers.md

.. toctree::
   :caption: Avaliable Filters
   :maxdepth: 1
   :glob:

   avaliable_filters/length_filters.md
   avaliable_filters/script_and_language_identification_filters.md
   avaliable_filters/special_character_and_similarity_filters.md
   avaliable_filters/language_model_filters.md
   avaliable_filters/alignment_model_filters.md
   avaliable_filters/sentence_embedding_filters.md
   avaliable_filters/custom_filters.md

.. toctree::
   :caption: Avaliable Preprocessors
   :maxdepth: 1

   avaliable_preprocessors/tokenizer.md
   avaliable_preprocessors/detokenizer.md
   avaliable_preprocessors/whitespaceNormalizer.md
   avaliable_preprocessors/reg_exp_sub.md
   avaliable_preprocessors/monolingual_sentence_splitter.md
   avaliable_preprocessors/bpe_segmentation.md
   avaliable_preprocessors/morfessor_segmentation.md
   avaliable_preprocessors/custom_preprocessors.md


.. toctree::
   :caption: Other information
   :maxdepth: 1
   
   other_tools.md
   CONTRIBUTING.md
   CHANGELOG.md