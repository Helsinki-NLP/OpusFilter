# Document scoring filters

## WDSFilter

Filter segments by their Web Docs Scorer
(<https://github.com/pablop16n/web-docs-scorer>) score. The Web Docs
Scorer was developed for scoring whole web documents consisting of
several newline-separated segments. In OpusFilter, each segment of an
input pair is treated as a document, and the newline-separated lines
of the segment are scored as the segments of the document. This is
useful with document-level data, where the segments may contain
embedded newlines, e.g. when using the JSONL format for the input
files.

**Requires installing optional libraries**: `docscorer` (GPL-3.0
license), see {doc}`../installation`. The Web Docs Scorer package has
no PyPI distribution, so the `[wds]` extra installs it directly from
GitHub.

Parameters:

* `ref_language`: reference language and script codes for the segments
  of each input pair, e.g. `['spa_latn']` for monolingual or
  `['fin_latn', 'spa_latn']` for bilingual pairs. The codes should be
  in the format `<ISO 639-3 language code>_<ISO 15924 script code>`
  (e.g. `spa_latn`). The language subscore of the Web Docs Scorer and
  its language-specific parameters are based on these codes.
* `threshold`: score threshold (default 0). A segment is accepted if
  its score is higher than the threshold. Can be either a scalar value
  applied to all segments or a list with one value per segment.
* `subscore`: score subscore to use instead of the overall Web Docs
  Scorer score (optional; default `null`). The available subscores are
  `language_score`, `url_score`, `punctuation_score`,
  `singular_chars_score`, `numbers_score`, `repeated_score`,
  `n_long_segments_score`, `great_segment_score`,
  `informativeness_score`, and `short_segments_score`. All scores are
  on the scale 0–1, and the higher the score, the better. Some of the
  subscores (e.g. `punctuation_score`) may be more informative than
  the overall score when the documents consist of only one or a few
  short segments.
* `lid_method`: language identification method used internally to
  generate the segment-level language codes for the Web Docs Scorer.
  Supported values: `fasttext` and `lingua`.
* `lid_options`: options for the language identification method
  (optional; default `{}`):
  * For `fasttext`: `model_path` pointing to a fasttext model that
    outputs labels in the `<language>_<script>` format, e.g. the
    OpenLID-v2 model for HPLT
    (<https://huggingface.co/laurievb/OpenLID-v2>).
  * For `lingua`: `lingua_mode` (`low` or `high`) and
    `langid_languages` for restricting the set of possible languages.
* `score_for_empty`: score for empty documents (default 1.0).

Example configurations:

```yaml
- WDSFilter:
    threshold: 0.8
    ref_language: [spa_latn]
    lid_method: fasttext
    lid_options: {model_path: lid201-model.bin}
```

```yaml
- WDSFilter:
    subscore: punctuation_score
    threshold: [0.5, 0.3]
    ref_language: [fin_latn, spa_latn]
    lid_method: lingua
    lid_options: {lingua_mode: high}
```

The `ref_language` codes give the expected language for each segment.
The segment-level language codes produced by the language
identification are compared against these reference codes to obtain
the `language_score` subscore. The language identification methods do
not necessarily output the `<language>_<script>` format directly. With
`fasttext`, labels that consist of only a language code are mapped to
the `<language>_<script>` format using the script information in the
Web Docs Scorer language family data. With `lingua`, the detected
language is mapped to an ISO 639-3 code and the script is looked up
from the same data. If a language has several possible scripts in the
language family data (e.g. Serbian can be written with both Latin and
Cyrillic scripts), the script in the `ref_language` code is assumed.

**Caveats:**

The Web Docs Scorer is meant for whole web documents, and several of
its subscores (e.g. `informativeness_score`, `repeated_score`, and the
two `long_segment` subscores) assume connections between the segments
of a document. When the documents consist of only a single short
segment, the overall score tends to get low values, and filtering on
the overall score may thus reject most of the data. The individual
subscores may be more useful in this case.

The `docscorer` package is distributed under the GPL-3.0 license and
pins `numpy>=2` and `scipy>=1.14`. Installing the `[wds]` extra can
therefore conflict with older versions of the other OpusFilter
dependencies (e.g. `pandas` or `scikit-learn`), and prevents combining
with the `[fasttext]` extra, as `fasttext` does not support
`numpy>=2`. The `docscorer` package actually works also with older
`numpy` versions; see {doc}`../installation` for installing it without
its dependencies when the fasttext language identification method is
needed.