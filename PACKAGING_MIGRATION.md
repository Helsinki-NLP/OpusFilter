# OpusFilter Packaging Migration Plan

## Overview
The package has been migrated from `setup.py` to a modern `pyproject.toml` configuration following PEP 517/518 and PEP 621 standards.

## Key Improvements Made

### 1. **Complete Migration to pyproject.toml**
- All metadata moved to `[project]` section following PEP 621
- Removed dependency on `setup.py` for build configuration
- Maintained backward compatibility with existing build system

### 2. **Enhanced Metadata**
- Added comprehensive classifiers including:
  - Development status
  - Intended audience  
  - Topic classifications
  - All supported Python versions (3.8-3.13)
- Added both authors and maintainers fields
- Added relevant keywords for better PyPI discoverability
- Added license file reference

### 3. **Improved Dependencies Management**
- Organized optional dependencies with proper extras
- Created "all" extra that includes all optional dependencies
- Maintained version constraints and Python version markers
- Separated core dependencies from optional ones

### 4. **Rich URL References**
- Added multiple project URLs:
  - Homepage
  - Documentation
  - Repository
  - Bug tracker
  - Changelog
  - Research paper
- Provides users with multiple entry points to project information

### 5. **Modern Entry Points Configuration**
- Migrated from `scripts` to `[project.scripts]`
- Uses module references instead of script paths
- Better integration with modern Python packaging tools

### 6. **Development Tools Configuration**
- Added configuration for common development tools:
  - pytest (test paths)
  - flake8 (extended ignore patterns)
  - pylint (disabled rules for opusfilter patterns)
  - black (line length)
  - isort (profile settings)

### 7. **Version Management**
- Kept `setuptools_scm` for version management
- Writes version to `opusfilter/_version.py`

## Remaining Tasks

### 1. **Update Script Entry Points**
The current configuration assumes scripts have `main()` functions. If the scripts don't have these functions, you need to either:

Option A: Add `main()` functions to each script:
```python
def main():
    # Current script content
    
if __name__ == "__main__":
    main()
```

Option B: Use the old script path approach:
```toml
[project.scripts]
opusfilter = "opusfilter.bin.opusfilter"
```

Option C: Create wrapper modules in `opusfilter/scripts/` and reference them.

### 2. **Add Version File Handling**
Add `opusfilter/_version.py` to `.gitignore` and ensure it's properly imported:
```python
# In opusfilter/__init__.py
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
```

### 3. **Update Package Data**
If there are data files that need to be included, add:
```toml
[tool.setuptools.package-data]
opusfilter = ["data/*.yaml", "data/*.json"]
```

### 4. **Validate Installation**
After migration, test:
- `pip install .`
- `pip install .[all]`
- `pip install -e .`
- Entry points are available: `opusfilter --help`

### 5. **Update Documentation**
- Update installation instructions to use `pyproject.toml`
- Remove references to `setup.py install`
- Ensure README.md is accurate

### 6. **CI/CD Updates**
Update GitHub Actions or other CI to:
- Use modern build tools (`pip install .`)
- Test with all Python versions specified in classifiers
- Validate package metadata

## Benefits of Migration

1. **PEP Compliance**: Full compliance with modern packaging standards
2. **Better Tooling Support**: Improved integration with pip, build, poetry, etc.
3. **Cleaner Configuration**: All metadata in one declarative file
4. **Enhanced Discoverability**: Better PyPI metadata
5. **Future-Proof**: Ready for upcoming packaging features
6. **Reduced Maintenance**: No need to maintain multiple config files

## Breaking Changes

- None for end users (API remains the same)
- Development workflow changes (use `pip install .` instead of `python setup.py install`)
- Build system fully standardized

## Testing the Migration

```bash
# Clean installation test
pip install --upgrade build
python -m build

# Test installation from wheel
pip install dist/opusfilter-*.whl

# Test entry points
opusfilter --help

# Run tests with all extras
pytest
```