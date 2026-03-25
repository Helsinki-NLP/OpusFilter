"""Tests for configuration validation."""

import unittest

from opusfilter import ConfigurationError
from opusfilter.validate import (
    ConfigValidator,
    validate_configuration,
    ValidationError,
    KNOWN_STEP_TYPES,
)


class TestConfigValidator(unittest.TestCase):
    """Test ConfigValidator class."""

    def test_valid_minimal_config(self):
        """Minimal valid config passes validation."""
        config = {
            'steps': [{'type': 'filter', 'parameters': {}}]
        }
        errors = ConfigValidator().validate(config)
        self.assertEqual(errors, [])

    def test_valid_config_with_common(self):
        """Config with common section passes."""
        config = {
            'common': {'output_directory': 'output'},
            'steps': [{'type': 'filter', 'parameters': {}}]
        }
        errors = ConfigValidator().validate(config)
        self.assertEqual(errors, [])

    def test_config_must_be_dict(self):
        """Non-dict configuration raises error."""
        errors = ConfigValidator().validate("not a dict")
        self.assertEqual(len(errors), 1)
        self.assertIn("must be a YAML dictionary", str(errors[0]))

    def test_missing_steps(self):
        """Missing steps section is allowed (for programmatic use)."""
        config = {'common': {'output_directory': 'out'}}
        errors = ConfigValidator().validate(config)
        self.assertEqual(errors, [])

    def test_steps_null(self):
        """Steps set to null is allowed but not validated."""
        config = {'steps': None}
        errors = ConfigValidator().validate(config)
        self.assertEqual(errors, [])

    def test_steps_must_be_list(self):
        """Non-list steps raises error."""
        config = {'steps': {'type': 'filter'}}
        errors = ConfigValidator().validate(config)
        self.assertEqual(len(errors), 1)
        self.assertIn("steps must be a list", str(errors[0]))

    def test_unknown_step_type(self):
        """Unknown step type raises error with suggestion."""
        config = {'steps': [{'type': 'fiter', 'parameters': {}}]}
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("Unknown step type 'fiter'" in str(e) for e in errors))
        suggestion = next(e for e in errors if "fiter" in str(e))
        self.assertIn("Did you mean 'filter'?", str(suggestion))

    def test_unknown_step_type_no_suggestion(self):
        """Unknown step type without close match has no suggestion."""
        config = {'steps': [{'type': 'xyzabc123', 'parameters': {}}]}
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("Unknown step type 'xyzabc123'" in str(e) for e in errors))

    def test_missing_type_field(self):
        """Missing type field raises error."""
        config = {'steps': [{'parameters': {}}]}
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("missing required 'type'" in str(e).lower() for e in errors))

    def test_missing_parameters_field(self):
        """Missing parameters field raises error."""
        config = {'steps': [{'type': 'filter'}]}
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("missing required 'parameters'" in str(e).lower() for e in errors))

    def test_unknown_common_key(self):
        """Unknown key in common raises error with suggestion."""
        config = {
            'common': {'output_directory': 'out', 'ouptut_dir': 'x'},
            'steps': [{'type': 'filter', 'parameters': {}}]
        }
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("Unknown key 'ouptut_dir'" in str(e) for e in errors))
        suggestion = next(e for e in errors if "ouptut_dir" in str(e))
        self.assertIn("Did you mean 'output_directory'?", str(suggestion))

    def test_unknown_step_key(self):
        """Unknown key in step raises error with suggestion."""
        config = {
            'steps': [{'type': 'filter', 'parameters': {}, 'dependson': []}]
        }
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("Unknown key 'dependson'" in str(e) for e in errors))
        suggestion = next(e for e in errors if "dependson" in str(e))
        self.assertIn("Did you mean 'depends_on'?", str(suggestion))

    def test_variables_must_be_lists(self):
        """Variable values that are not lists raise error."""
        config = {
            'steps': [{
                'type': 'filter',
                'parameters': {},
                'variables': {'lang': 'de'}
            }]
        }
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("must define a list" in str(e) for e in errors))

    def test_variables_inconsistent_lengths(self):
        """Variables with inconsistent lengths raise error."""
        config = {
            'steps': [{
                'type': 'filter',
                'parameters': {},
                'variables': {'lang': ['de', 'en'], 'pair': ['de-en']}
            }]
        }
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("inconsistent lengths" in str(e) for e in errors))

    def test_variables_empty_list(self):
        """Variable with empty list raises error."""
        config = {
            'steps': [{
                'type': 'filter',
                'parameters': {},
                'variables': {'lang': []}
            }]
        }
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("empty value list" in str(e) for e in errors))

    def test_depends_on_must_be_list(self):
        """Non-list depends_on raises error."""
        config = {
            'steps': [{
                'type': 'filter',
                'parameters': {},
                'depends_on': 'not a list'
            }]
        }
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("depends_on must be a list" in str(e) for e in errors))

    def test_depends_on_must_be_strings(self):
        """Non-string values in depends_on raise error."""
        config = {
            'steps': [{
                'type': 'filter',
                'parameters': {},
                'depends_on': ['file.gz', 123]
            }]
        }
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("list of filenames" in str(e) for e in errors))

    def test_step_must_be_dict(self):
        """Non-dict step raises error."""
        config = {'steps': ['not a dict']}
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("must be a dictionary" in str(e) for e in errors))

    def test_common_must_be_dict(self):
        """Non-dict common raises error."""
        config = {'common': 'not a dict', 'steps': []}
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("common must be a dictionary" in str(e) for e in errors))

    def test_constants_must_be_dict(self):
        """Non-dict constants raises error."""
        config = {
            'common': {'constants': 'not a dict'},
            'steps': [{'type': 'filter', 'parameters': {}}]
        }
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("constants must be a dictionary" in str(e) for e in errors))

    def test_output_directory_must_be_string(self):
        """Non-string output_directory raises error."""
        config = {
            'common': {'output_directory': 123},
            'steps': [{'type': 'filter', 'parameters': {}}]
        }
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("output_directory must be a string" in str(e) for e in errors))

    def test_chunksize_must_be_positive_int(self):
        """Invalid chunksize raises error."""
        config = {
            'common': {'chunksize': -5},
            'steps': [{'type': 'filter', 'parameters': {}}]
        }
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("chunksize must be a positive integer" in str(e) for e in errors))

    def test_default_n_jobs_must_be_positive_int(self):
        """Invalid default_n_jobs raises error."""
        config = {
            'common': {'default_n_jobs': 0},
            'steps': [{'type': 'filter', 'parameters': {}}]
        }
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("default_n_jobs must be a positive integer" in str(e) for e in errors))

    def test_multiple_errors(self):
        """Multiple errors are collected."""
        config = {
            'common': {'ouptut': 'x', 'invalid': 'y'},
            'steps': [
                {'type': 'fiter', 'parameters': {}},
                {'type': 'filter'},
            ]
        }
        errors = ConfigValidator().validate(config)
        self.assertGreater(len(errors), 3)

    def test_all_known_step_types_valid(self):
        """All known step types pass validation."""
        for step_type in KNOWN_STEP_TYPES:
            config = {'steps': [{'type': step_type, 'parameters': {}}]}
            errors = ConfigValidator().validate(config)
            self.assertEqual(errors, [], f"Step type '{step_type}' should be valid")

    def test_step_type_must_be_string(self):
        """Non-string type raises error."""
        config = {'steps': [{'type': 123, 'parameters': {}}]}
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("type must be a string" in str(e) for e in errors))

    def test_parameters_must_be_dict(self):
        """Non-dict parameters raises error."""
        config = {'steps': [{'type': 'filter', 'parameters': []}]}
        errors = ConfigValidator().validate(config)
        self.assertTrue(any("parameters must be a dictionary" in str(e) for e in errors))


class TestValidateConfiguration(unittest.TestCase):
    """Test validate_configuration function."""

    def test_valid_config_no_exception(self):
        """Valid config does not raise."""
        config = {
            'steps': [{'type': 'filter', 'parameters': {}}]
        }
        validate_configuration(config)

    def test_invalid_config_raises_configuration_error(self):
        """Invalid config raises ConfigurationError."""
        config = {'common': 'not a dict', 'steps': []}
        with self.assertRaises(ConfigurationError) as ctx:
            validate_configuration(config)
        self.assertIn("Configuration validation failed", str(ctx.exception))

    def test_error_message_format(self):
        """Error message includes all validation errors."""
        config = {'steps': [], 'common': 'not a dict'}
        with self.assertRaises(ConfigurationError) as ctx:
            validate_configuration(config)
        error_str = str(ctx.exception)
        self.assertIn("common must be a dictionary", error_str)
        self.assertIn("Configuration validation failed", error_str)


class TestValidationError(unittest.TestCase):
    """Test ValidationError class."""

    def test_str_with_path(self):
        """String representation includes path."""
        err = ValidationError("steps[0].type", "Unknown step type")
        self.assertEqual(str(err), "steps[0].type: Unknown step type")

    def test_str_with_suggestion(self):
        """String representation includes suggestion."""
        err = ValidationError(
            "steps[0].type",
            "Unknown step type 'fiter'",
            "Did you mean 'filter'?"
        )
        self.assertIn("Did you mean 'filter'?", str(err))

    def test_str_without_path(self):
        """String representation without path."""
        err = ValidationError("", "Configuration must be a YAML dictionary")
        self.assertEqual(str(err), "Configuration must be a YAML dictionary")


class TestSuggestSimilar(unittest.TestCase):
    """Test the suggestion algorithm."""

    def test_suggestion_for_close_match(self):
        """Suggestion provided for close match."""
        suggestion = ConfigValidator._suggest_similar("filetr", KNOWN_STEP_TYPES)
        self.assertEqual(suggestion, "Did you mean 'filter'?")

    def test_no_suggestion_for_different_word(self):
        """No suggestion for very different word."""
        suggestion = ConfigValidator._suggest_similar("xyz", KNOWN_STEP_TYPES)
        self.assertIsNone(suggestion)

    def test_no_suggestion_for_short_word(self):
        """No suggestion for short words."""
        suggestion = ConfigValidator._suggest_similar("fi", KNOWN_STEP_TYPES)
        self.assertIsNone(suggestion)


if __name__ == '__main__':
    unittest.main()
