# Using score files

## join

Join two or more score files.

Parameters:

* `inputs`: input files containing scores in JSON Lines format
* `output`: output file for joined scores
* `keys`: a list containing dictionary keys for each input file (optional; default `null`)

If the list of keys is provided, the input objects are inserted under
the corresponding key. The objects can also be inserted deeper in a
hierarchical score dictionary by using a key that has dot-separated
parts. For example, `x.y` means setting key `y` under the key `x`. If
the keys are not provided, or the key corresponding to the input file
is `null`, output object will be updated with the input object and
existing keys will be overwritten.

For example, if you have scores for the source and target sentences
created by external tools (`myscores.src` and `myscores.tgt`
containing one number per line), and you want to join them with an
existing score file created by OpusFilter (`scores.jsonl.gz`), you can
do it like this:

```
  - type: join
    parameters:
      inputs:
      - scores.jsonl.gz
      - myscores.src
      - myscores.tgt
      keys:
      - null
      - MyScore.src
      - MyScore.tgt
      output: scores-joined.jsonl.gz
```

Apart from the old scores from `scores.jsonl.gz`, each line should now
contain `{"MyScore": {"src": ..., "tgt": ...}}`.

## sort

Sort files based on score values.

Parameters:

* `inputs`: input files to sort
* `outputs`: sorted output files
* `values`: input file for values used in sorting
* `reverse`: `true` for descending sort (optional; default `false`)
* `key`: if values file contain JSON objects, use the key to select field (optional; default `null`)
* `type`: force type conversion for the value (optional; `float`, `int`, `str`, or default `null`)

The values file should contain one JSON object per line. If a line
cannot be interpreted as a JSON object, it is read as a plain unicode
string. Dots (`.`) in the key are interpreted as multiple get operations
(e.g. `x.y` expects that there is key `x` under the key `y`). List items
can be accessed with integer keys. The type conversion can be used e.g.
for forcing numerical values to be compared as strings.