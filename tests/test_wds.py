import logging
import os
import shutil
import tempfile
import unittest

from iso639 import Lang

from opusfilter import ConfigurationError
from opusfilter.util import file_download
from opusfilter.wds import SUBSCORES, WDSFilter

try:
    import docscorer  # noqa: F401
except ImportError:
    logging.warning("Could not import docscorer")

try:
    import fasttext  # noqa: F401
except ImportError:
    logging.warning("Could not import fasttext")


class TestWDSFilterConfig(unittest.TestCase):

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_missing_ref_language(self):
        with self.assertRaises(ConfigurationError):
            WDSFilter(lid_method="lingua")

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_invalid_ref_language(self):
        for code in ["spa", "spalatn", "spa_LATN", "spa-xx", "spa_latn_extra"]:
            with self.assertRaises(ConfigurationError):
                WDSFilter(ref_language=[code], lid_method="lingua")

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_invalid_subscore(self):
        with self.assertRaises(ConfigurationError):
            WDSFilter(ref_language=["spa_latn"], lid_method="lingua", subscore="foo")

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_missing_lid_method(self):
        with self.assertRaises(ConfigurationError):
            WDSFilter(ref_language=["spa_latn"])

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_invalid_lid_method(self):
        with self.assertRaises(ConfigurationError):
            WDSFilter(ref_language=["spa_latn"], lid_method="cld2")

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_missing_fasttext_model(self):
        with self.assertRaises(ConfigurationError):
            WDSFilter(ref_language=["spa_latn"], lid_method="fasttext")

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_unrecognized_lid_options(self):
        with self.assertRaises(ConfigurationError):
            WDSFilter(ref_language=["spa_latn"], lid_method="lingua", lid_options={"foo": 1})

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_invalid_lingua_mode(self):
        with self.assertRaises(ConfigurationError):
            WDSFilter(ref_language=["spa_latn"], lid_method="lingua", lid_options={"lingua_mode": "mid"})

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_invalid_threshold(self):
        with self.assertRaises(ConfigurationError):
            WDSFilter(ref_language=["spa_latn"], lid_method="lingua", threshold="high")

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_threshold_mismatch(self):
        with self.assertRaises(ConfigurationError):
            WDSFilter(ref_language=["spa_latn"], lid_method="lingua", threshold=[0.5, 0.6])

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_valid_config(self):
        model = WDSFilter(ref_language=["fin_latn", "spa_latn"], threshold=0.5, lid_method="lingua")
        self.assertTrue(all(model.thresholds[idx] == 0.5 for idx in range(len(model.ref_language))))

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_threshold_list_config(self):
        model = WDSFilter(ref_language=["fin_latn", "spa_latn"], threshold=[0.5, 0.3], lid_method="lingua")
        self.assertEqual(model.thresholds, [0.5, 0.3])


class TestWDSFilterLangScript(unittest.TestCase):

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_lingua_lang_script(self):
        model = WDSFilter(ref_language=["spa_latn"], lid_method="lingua", lid_options={"langid_languages": ["es", "en"]})
        result = model._lang_script(
            "Esta es una frase española bastante larga que contiene muchas palabras y letras", "spa_latn"
        )
        self.assertEqual(result, "spa_latn")
        self.assertEqual(Lang(result.split("_")[0]).pt1, "es")

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_lang_script_empty(self):
        model = WDSFilter(ref_language=["spa_latn"], lid_method="lingua")
        self.assertEqual(model._lang_script("", "spa_latn"), "spa_latn")
        self.assertEqual(model._lang_script("\n", "spa_latn"), "spa_latn")

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_lang_script_wrong_language(self):
        model = WDSFilter(ref_language=["spa_latn"], lid_method="lingua", lid_options={"langid_languages": ["es", "en"]})
        result = model._lang_script("This is an English sentence that clearly identifies as English language text", "spa_latn")
        self.assertEqual(result.split("_")[0], Lang("en").pt3)


class TestWDSFilterScore(unittest.TestCase):

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_language_subscore_accept(self):
        model = WDSFilter(
            ref_language=["spa_latn"],
            subscore="language_score",
            threshold=0.5,
            lid_method="lingua",
            lid_options={"langid_languages": ["es", "en"]},
        )
        pairs = [
            (
                "Esta es una frase española bastante larga que contiene muchas palabras y letras "
                "distintas para el identificador",
            ),
            ("This is an English sentence with lots of words that clearly identifies " "as English language overall",),
        ]
        pair_expecteds = [True, False]
        for pair_score, pair_expected in zip(model.score(pairs), pair_expecteds):
            self.assertEqual(model.accept(pair_score), pair_expected)
            self.assertTrue(len(pair_score) == 1)

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_score_bilingual(self):
        model = WDSFilter(
            ref_language=["fin_latn", "spa_latn"], subscore="punctuation_score", threshold=[0.5, 0.3], lid_method="lingua"
        )
        pairs = [
            (
                "Tämä on suomenkielinen lause, joka on aivan tarpeeksi pitkä näytettäväksi tässä testissä.",
                "Esta es una frase española que tiene la suficiente longitud para ser evaluada.",
            ),
        ]
        for pair_score in model.score(pairs):
            self.assertTrue(len(pair_score) == 2)

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_empty_document(self):
        model = WDSFilter(ref_language=["spa_latn"], lid_method="lingua", score_for_empty=0.9)
        self.assertEqual(model._score_document("", "spa_latn"), 0.9)
        self.assertEqual(model._score_document("\n\n", "spa_latn"), 0.9)

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_pair_length_mismatch(self):
        model = WDSFilter(ref_language=["spa_latn"], lid_method="lingua")
        with self.assertRaises(ValueError):
            list(model.score([("a", "b")]))

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_subscore_map(self):
        self.assertEqual(SUBSCORES["language_score"], 1)
        self.assertEqual(SUBSCORES["punctuation_score"], 3)
        self.assertEqual(SUBSCORES["short_segments_score"], 10)

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_overall_score_accept(self):
        model = WDSFilter(ref_language=["spa_latn"], threshold=0, lid_method="lingua")
        scores = list(
            model.score(
                [
                    (
                        "Esta es una frase española bastante larga que contiene muchas palabras "
                        "y letras distintas para el identificador",
                    )
                ]
            )
        )
        self.assertEqual(len(scores), 1)
        self.assertTrue(all(isinstance(value, float) for value in scores[0]))


@unittest.skipIf("fasttext" not in globals(), "fasttext not installed")
class TestWDSFilterFasttext(unittest.TestCase):

    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

    @classmethod
    def setUpClass(self):
        if "docscorer" not in globals():
            return
        self.tempdir = tempfile.mkdtemp()
        self.testmodel = os.path.join(self.tempdir, "model.ftz")
        try:
            file_download(self.model_url, self.testmodel)
        except Exception:
            self.testmodel = None
        else:
            self.model = WDSFilter(
                ref_language=["spa_latn"], lid_method="fasttext", lid_options={"model_path": self.testmodel}
            )

    @classmethod
    def tearDownClass(self):
        if hasattr(self, "tempdir"):
            shutil.rmtree(self.tempdir)

    @unittest.skipIf("docscorer" not in globals(), "docscorer not installed")
    def test_fasttext_lang_script(self):
        if not hasattr(self, "model"):
            self.skipTest("Failed to download test resources")
        result = self.model._lang_script(
            "Esta es una frase española bastante larga que contiene muchas palabras y letras", "spa_latn"
        )
        self.assertRegex(result, r"^[a-z]{3}_[a-z]{4}$")
        english = self.model._lang_script("This is an English sentence with lots of words", "spa_latn")
        self.assertEqual(english.split("_")[0], Lang("en").pt3)


if __name__ == "__main__":
    unittest.main()
