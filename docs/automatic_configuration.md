# Automatic configuration generation

You can generate OpusFilter config files with the `opusfilter-autogen`
script. Currently the script supports only adding a single filter
step, with a few options for determining the filter parameters.

The usage description for the script is as follows:
```text
usage: opusfilter-autogen [-h] --files TEXTFILE [TEXTFILE ...]
                          [--langs LANGCODE [LANGCODE ...]] [--scripts SCRIPT [SCRIPT ...]]
                          [--filter-params {cluster,default,percentiles}]
                          [--sample-size SAMPLE_SIZE] [--noisy-percentile NOISY_PERCENTILE]
                          [--work-dir WORK_DIR] [--inter-dir INTER_DIR] [--plot]
                          [--overwrite] [-o CONFIGFILE]

Generate initial configuration based on parallel text data

options:
  -h, --help            show this help message and exit
  --files TEXTFILE [TEXTFILE ...]
                        parallel text input file(s)
  --langs LANGCODE [LANGCODE ...]
                        Language codes corresponding to the input files. If omitted,
                        LanguageIDFilters will not be used.
  --scripts SCRIPT [SCRIPT ...]
                        Alphabetic scripts (e.g. Latin) corresponding to the input files.
                        If omitted, CharacterScoreFilter will not be used.
  --filter-params {default,percentiles,unsupervised}
                        Method for selecting filter parameters (default: unsupervised)
  --sample-size SAMPLE_SIZE
                        Max number of sentence pairs used for clustering (default 100000)
  --noisy-percentile NOISY_PERCENTILE
                        Proportion of the data considered to be noisy; only for percentiles
                        method (default 0.001)
  --work-dir WORK_DIR   Location of the source and target files for the generated
                        configuration (default work)
  --inter-dir INTER_DIR
                        Save intermediate files in this directory (use a temporary
                        directory if not given)
  --plot                Show a scatter plot of the clustering and histograms of feature
                        data distributions
  --overwrite           Overwrite existing config file and intermediate files
  -o CONFIGFILE, --output CONFIGFILE
                        Output configuration file (default -)
```

The `--filter-params` options sets how the filter parameters are set.
The option `default` uses the default parameters defined in the filter
classes. The option `percentiles` assumes that a proportion of the
data (set by `--noisy-percentile`) is noisy, and sets the thresholds
for each filter independently based on the percentile. The
`unsupervised` option is likely the most useful of the three, and
described in more detail below.

## Unsupervised feature selection for filters

This implements the method introduced by {cite:t}`aulamo-etal-2023-unsupervised`.
It takes a parallel corpus as an input and tries to separate the clean
and noisy samples to generate threshold parameters for filters. The
currently supported filters are `AlphabetRatioFilter`,
`CharacterScoreFilter`, `LanguageIDFilter`, `LengthRatioFilter`,
`NonZeroNumeralsFilter` and `TerminalPunctuationFilter`, but this list
will be expanded and made more flexible in the future.

First, we remove duplicates and empty sentences from the input
corpus. Next, we take a subset (`--sample-size`, 100k sentence pairs
by default) of the corpus and produce scores for each sentence pair in
the subset with the previously mentioned filters. These scores are
used as features for K-means clustering to classify the sentence pairs
into clean and noisy pairs. The values of the noisy cluster center are
used as the filter threshold parameters in the generated config file.

Figures from the clustering and score histograms are plotted given the
`--plot` option. If you want also to save the intermediate files, make
sure to use the `--inter-dir` argument.
