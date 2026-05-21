# Clustering for curriculum learning

Given a score file created with the `score` command, you can use the
`opusfilter-curriculum` script to cluster the scored data into buckets. The
output is a label file that contains a bucket label for each line in the score
file. The labels are integers that indicate the cleanness order of the buckets.
The integer 0 represents the cleanest bucket, and the higher the integer is,
the noisier the bucket is.

The usage description for the script is as follows:
```text
usage: opusfilter-curriculum [-h] --scores SCOREFILE --data-size INT
                             [--method {babystep}] [--sample-size INT]
                             [--clusters INT] [--data_inc INT] [--gmean]
                             [--seed SEED] [--work-dir WORK_DIR]
                             [--chunksize CHUNKSIZE] [--overwrite]
                             [-o OUTPUTFILE]

Create a curriculum learning schedule for a dataset

options:
  -h, --help            show this help message and exit
  --scores SCOREFILE    jsonl score file
  --data-size INT       Number of sentence pairs in training set (to calculate
                        number of buckets)
  --method {babystep}   Curriulum method (default: babystep)
  --sample-size INT     Max number of sentence pairs used for data-based methods
                        (default 100000)
  --clusters INT, -k INT
                        Number of clusters for Kmeans clustering for the first b
                        sentence pairs (default 10)
  --data_inc INT, -b INT
                        Number of clusters is k + (data_size-b)/b
                        (default 100000)
  --gmean               Use geometric mean instead of arithmetic mean when
                        comparing cluster centers
  --seed SEED           Seed for subset
  --work-dir WORK_DIR   Location of the source and target files (default work)
  --chunksize CHUNKSIZE
                        Chunksize during Kmeans prediction (default 500000)
  --overwrite           Overwrite existing intermediate files
  -o OUTPUTFILE, --output OUTPUTFILE
                        Output file (default -)
```

The `--scores` options takes the score file as an input, and the `--data-size`
option takes in the number of lines in the score file. The bucket labels are
written into the output file defined with the `--output` option.

The clustering method is introduced by {cite:t}`aulamo-etal-2026-challenge`.
First, a sample is taken from the score file (100k by default, can be changed
with the `--sample-size` option). The sample data is clustered with K-means into
buckets. The number of clusters is determined with the formula k+(data_size-b)/b
where data_size (`--data_size`) is the number of sentence pairs in the dataset,
k (`--clusters`) is the number of clusters for the first b sentence pairs, and
b (`--data_inc`) is the average cluster size. Once the sample data has been
clustered with K-means, all sentence pairs in the full dataset are labeled 
according to the trained clustering. The buckets are sorted into cleanness
order based on the mean noise values of the cluster centers. The buckets can be
used for curriculum learning in NMT, as is done by {cite:t}`aulamo-etal-2026-challenge`.
