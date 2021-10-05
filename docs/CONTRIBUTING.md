Questions, bug reports, and feature wishes are welcome in the GitHub
issues page. We are also happy to consider pull requests. There are a
few rules for pull requests:

* Make a pull request to the `develop` branch instead of `master`.
* The code should support at least Python versions from 3.6 to 3.8.
* Please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/). Exception: The maximum line length is 127 characters instead of 79.
* Especially for new features, please include test cases for unit testing.

PEP 8 compatibility can be checked with `flake8`. Install it e.g. via
`pip` and run `flake8 opusfilter/` in the project root.

The unit tests are located in the `tests` directory. To run them,
install [pytest](https://pytest.org/) and run `pytest tests/` in the
project root. (Also [nosetests](https://nose.readthedocs.io/) should
work, if you have VariKN and eflomal set up as instructed - `pytest`
skips the respective tests if not.)

GitHub workflows defined in the project run automatically `flake8`
checks and unit testing with `pytest` using Python 3.6, 3.7, and 3.8.

Especially for larger contributions, consider using a code analysis
tool like [Pylint](https://github.com/PyCQA/pylint). Install it
e.g. via `pip`, run `pylint opusfilter/` in the project root and fix
at least everything that is simple to fix in the new code (note that
the current code yields a few warnings from `pylint`).
