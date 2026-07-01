"""Configuration validation for OpusFilter."""

from typing import Any, Dict, List, Optional, Set

from rapidfuzz.distance.Levenshtein import distance as levenshtein_distance

from . import ConfigurationError


KNOWN_STEP_TYPES: Set[str] = {
    'opus_read', 'hf_read', 'filter', 'concatenate', 'subset', 'train_bpe',
    'train_morfessor', 'train_ngram', 'train_alignment', 'train_nearest_neighbors',
    'score', 'train_classifier', 'classify', 'join', 'sort', 'head', 'tail',
    'slice', 'product', 'remove_duplicates', 'split', 'unzip', 'preprocess',
    'download', 'write'
}

KNOWN_COMMON_KEYS: Set[str] = {
    'output_directory', 'chunksize', 'default_n_jobs', 'constants', 'slurm'
}

KNOWN_STEP_KEYS: Set[str] = {
    'type', 'parameters', 'variables', 'constants', 'depends_on'
}


class ValidationError:
    """Configuration validation error with context."""

    def __init__(
        self,
        path: str,
        message: str,
        suggestion: Optional[str] = None
    ):
        self.path = path
        self.message = message
        self.suggestion = suggestion

    def __str__(self) -> str:
        result = f"{self.path}: {self.message}" if self.path else self.message
        if self.suggestion:
            result += f" {self.suggestion}"
        return result


class ConfigValidator:
    """Validates OpusFilter YAML configuration structure."""

    def __init__(self):
        self.errors: List[ValidationError] = []

    def validate(self, configuration: Dict[str, Any]) -> List[ValidationError]:
        """Validate configuration and return list of errors."""
        self.errors = []

        if not isinstance(configuration, dict):
            self.errors.append(ValidationError(
                "", "Configuration must be a YAML dictionary"
            ))
            return self.errors

        self._validate_common(configuration.get('common'), 'common')
        if configuration.get('steps') is not None:
            self._validate_steps(configuration.get('steps'), 'steps')

        return self.errors

    def _validate_common(
        self,
        common: Any,
        path: str
    ) -> None:
        """Validate common section."""
        if common is None:
            return

        if not isinstance(common, dict):
            self.errors.append(ValidationError(
                path, "common must be a dictionary"
            ))
            return

        for key in common:
            if key not in KNOWN_COMMON_KEYS:
                suggestion = self._suggest_similar(key, KNOWN_COMMON_KEYS)
                self.errors.append(ValidationError(
                    f"{path}.{key}",
                    f"Unknown key '{key}' in common section",
                    suggestion
                ))

        if 'output_directory' in common:
            val = common['output_directory']
            if not isinstance(val, str):
                self.errors.append(ValidationError(
                    f"{path}.output_directory",
                    f"output_directory must be a string, got {type(val).__name__}"
                ))

        if 'chunksize' in common:
            val = common['chunksize']
            if not isinstance(val, int) or val <= 0:
                self.errors.append(ValidationError(
                    f"{path}.chunksize",
                    f"chunksize must be a positive integer, got {val}"
                ))

        if 'default_n_jobs' in common:
            val = common['default_n_jobs']
            if not isinstance(val, int) or val < 1:
                self.errors.append(ValidationError(
                    f"{path}.default_n_jobs",
                    f"default_n_jobs must be a positive integer, got {val}"
                ))

        if 'constants' in common and not isinstance(common['constants'], dict):
            self.errors.append(ValidationError(
                f"{path}.constants", "constants must be a dictionary"
            ))

    def _validate_steps(
        self,
        steps: Any,
        path: str
    ) -> None:
        """Validate steps section."""
        if steps is None:
            self.errors.append(ValidationError(
                path, "Missing required 'steps' section"
            ))
            return

        if not isinstance(steps, list):
            self.errors.append(ValidationError(
                path, "steps must be a list"
            ))
            return

        for i, step in enumerate(steps):
            self._validate_step(step, i, f"{path}[{i}]")

    def _validate_step(
        self,
        step: Any,
        index: int,
        path: str
    ) -> None:
        """Validate individual step."""
        if not isinstance(step, dict):
            self.errors.append(ValidationError(
                path, f"Step {index + 1} must be a dictionary"
            ))
            return

        for key in step:
            if key not in KNOWN_STEP_KEYS:
                suggestion = self._suggest_similar(key, KNOWN_STEP_KEYS)
                self.errors.append(ValidationError(
                    f"{path}.{key}",
                    f"Unknown key '{key}' in step",
                    suggestion
                ))

        if 'type' not in step:
            self.errors.append(ValidationError(
                path, f"Step {index + 1} is missing required 'type' field"
            ))
        else:
            step_type = step['type']
            if not isinstance(step_type, str):
                self.errors.append(ValidationError(
                    f"{path}.type",
                    f"step type must be a string, got {type(step_type).__name__}"
                ))
            elif step_type not in KNOWN_STEP_TYPES:
                suggestion = self._suggest_similar(step_type, KNOWN_STEP_TYPES)
                self.errors.append(ValidationError(
                    f"{path}.type",
                    f"Unknown step type '{step_type}'",
                    suggestion
                ))

        if 'parameters' not in step:
            self.errors.append(ValidationError(
                path, f"Step {index + 1} is missing required 'parameters' field"
            ))
        elif not isinstance(step.get('parameters'), dict):
            self.errors.append(ValidationError(
                f"{path}.parameters",
                f"parameters must be a dictionary, got {type(step['parameters']).__name__}"
            ))

        if 'variables' in step:
            self._validate_variables(step['variables'], path, index + 1)

        if 'constants' in step and not isinstance(step['constants'], dict):
            self.errors.append(ValidationError(
                f"{path}.constants", "constants must be a dictionary"
            ))

        if 'depends_on' in step:
            val = step['depends_on']
            if not isinstance(val, list):
                self.errors.append(ValidationError(
                    f"{path}.depends_on", "depends_on must be a list"
                ))
            elif not all(isinstance(x, str) for x in val):
                self.errors.append(ValidationError(
                    f"{path}.depends_on",
                    "depends_on must be a list of filenames (strings)"
                ))

    def _validate_variables(
        self,
        variables: Any,
        path: str,
        step_num: int
    ) -> None:
        """Validate variables section."""
        if not isinstance(variables, dict):
            self.errors.append(ValidationError(
                f"{path}.variables", "variables must be a dictionary"
            ))
            return

        lengths: Set[int] = set()
        for key, value in variables.items():
            if not isinstance(value, list):
                self.errors.append(ValidationError(
                    f"{path}.variables.{key}",
                    f"Variable '{key}' must define a list of values"
                ))
            elif not value:
                self.errors.append(ValidationError(
                    f"{path}.variables.{key}",
                    f"Variable '{key}' has empty value list"
                ))
            else:
                lengths.add(len(value))

        if len(lengths) > 1:
            self.errors.append(ValidationError(
                f"{path}.variables",
                f"Variables have inconsistent lengths: {sorted(lengths)}. "
                "All variables must have the same number of values"
            ))

    @staticmethod
    def _suggest_similar(word: str, options: Set[str]) -> Optional[str]:
        """Suggest similar option for misspelled word using Levenshtein distance."""
        if len(word) < 4:
            return None
        word_lower = word.lower()
        best_match = None
        best_score = float('inf')
        for option in options:
            if len(option) < 4:
                continue
            option_lower = option.lower()
            distance = levenshtein_distance(word_lower, option_lower)
            if distance < best_score:
                best_score = distance
                best_match = option
        max_distance = max(3, int(len(word) * 0.8))
        if best_match and best_score <= max_distance:
            return f"Did you mean '{best_match}'?"
        return None


def validate_configuration(configuration: Dict[str, Any]) -> None:
    """Validate configuration and raise ConfigurationError if invalid."""
    validator = ConfigValidator()
    errors = validator.validate(configuration)

    if errors:
        error_messages = [str(err) for err in errors]
        raise ConfigurationError(
            "Configuration validation failed:\n  - " +
            "\n  - ".join(error_messages)
        )
