# OpusFilter Development Guide

This guide contains essential information for agents working on the OpusFilter codebase.

## Project Overview

OpusFilter is a tool for filtering and processing parallel corpora. It's written in Python and supports filtering through various methods including language identification, language models, word alignment, and custom filters.

## Development Commands

### Testing
- **Run all tests**: `pytest tests/`
- **Run single test file**: `pytest tests/test_filename.py`
- **Run specific test**: `pytest tests/test_filename.py::ClassName::test_method`
- **Run with verbose output**: `pytest -v tests/`

### Linting and Code Quality
- **Check style**: `flake8 opusfilter/`
- **Run flake8 with statistics**: `flake8 opusfilter --count --exit-zero --statistics`
- **Run full linting (recommended for contributions)**: `pylint opusfilter/`
- **Format code**: `black opusfilter/ opusfilter/cli/`
- **Sort imports**: `isort opusfilter/ opusfilter/cli/`
- **Type checking**: `mypy opusfilter/ opusfilter/cli/`

### Environment Setup
- **Install with test dependencies**: `pip install .[test]`
- **Install optional dependencies for full functionality**: `pip install .[test,eflomal,jieba,mecab,varikn]`

## Code Style Guidelines

### General Rules
- Follow PEP 8 with a maximum line length of 127 characters (not 79)
- Support Python 3.8 to 3.13
- Use type hints where appropriate (see existing code for patterns)
- Maintain backward compatibility when possible

### Import Organization
```python
# Standard library imports
import os
import logging
from typing import Iterator, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import regex

# Local imports
from . import FilterABC, ConfigurationError
from .util import file_open, count_lines
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `LengthFilter`, `CrossEntropyFilter`)
- **Functions/Methods**: snake_case (e.g., `get_length`, `score_pairs`)
- **Variables**: snake_case (e.g., `min_length`, `accept_threshold`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `CLEAN_LOW`, `CLEAN_HIGH`)
- **Private members**: Leading underscore (e.g., `_internal_method`)

### Error Handling
- Use custom exception classes from `opusfilter.__init__`:
  - `OpusFilterError`: Base exception
  - `ConfigurationError`: For configuration-related errors
  - `OpusFilterRuntimeError`: For runtime errors
- Always include descriptive error messages
- Use logging for warnings and debug information

### Filter Implementation Pattern
When creating new filters, follow this pattern:
```python
class NewFilter(FilterABC):
    """Brief description of the filter"""
    
    score_direction = CLEAN_LOW  # or CLEAN_HIGH, CLEAN_BETWEEN, etc.
    accept_threshold = <value>
    reject_threshold = <value>
    
    def __init__(self, required_param, optional_param=default, **kwargs):
        # Validate parameters
        self.required_param = required_param
        self.optional_param = optional_param
        super().__init__(**kwargs)
    
    def score(self, pairs):
        """Yield scores for each sentence pair"""
        for pair in pairs:
            # Calculate score
            yield score
    
    def accept(self, score):
        """Return True if score passes the filter"""
        return <condition>
```

### Testing Guidelines
- Write unit tests for all new functionality
- Use descriptive test method names
- Test both success and failure cases
- Use `unittest.TestCase` as base class
- Mark tests that require optional dependencies with appropriate skips

### Logging
- Use module-level logger: `logger = logging.getLogger(__name__)`
- Log important operations and warnings
- Avoid excessive debug logging in production code

### File I/O
- Use `file_open()` from `opusfilter.util` for handling compressed files
- Support common formats: plain text, gzip, bzip2, lzma
- Use context managers for file operations

### Configuration
- Filters receive configuration through `__init__` parameters
- Use `**kwargs` to capture extra parameters and warn about them
- Validate parameters and raise `ConfigurationError` for invalid values

### Type Hints
- Use type hints for public APIs
- Import common types from `typing` module
- Follow existing patterns for complex types

### Documentation
- Add docstrings to all public classes and methods
- Use Google-style or NumPy-style docstrings
- Include examples in docstrings when helpful

## Project Structure

- `opusfilter/`: Main package directory
  - `filters.py`: Core filter implementations
  - `preprocessors.py`: Text preprocessing functions
  - `lm.py`: Language model filters
  - `lid.py`: Language identification filters
  - `embeddings.py`: Sentence embedding filters
  - `word_alignment.py`: Word alignment filters
  - `util.py`: Utility functions and helpers
  - `pipeline.py`: Processing pipeline implementation
  - `opusfilter.py`: Main application entry point
- `tests/`: Unit tests
- `docs/`: Documentation source files
- `work/`: Example configurations and working files

## Common Patterns

### Iterating over sentence pairs
```python
for pair in pairs:
    # pair is a list/tuple of strings
    # pair[0] is source, pair[1] is target, etc.
```

### Yielding scores
```python
def score(self, pairs):
    for pair in pairs:
        # Calculate score(s)
        yield score_value  # or yield [score1, score2, ...]
```

### Checking filter acceptance
```python
def accept(self, score):
    if isinstance(score, list):
        # Multiple scores
        return all(condition(s) for s in score)
    else:
        # Single score
        return condition(score)
```

## Git Workflow
- Make pull requests to the `develop` branch (not `master`)
- Include tests for new features
- Run linting and tests before submitting
- Follow conventional commit messages if possible
