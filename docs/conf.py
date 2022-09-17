# General information about the project.
project = "OpusFilter"
author = "Helsinki-NLP"
language = "en"

version = "latest"  # The short X.Y version.
# The full version, including alpha/beta/rc tags.
release = "latest"

extensions = [
    "myst_parser",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
]

source_suffix = [".rst", ".md"]
exclude_patterns = ["README.md"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {}
html_show_sourcelink = False
html_show_copyright = False
html_show_sphinx = False

suppress_warnings = ["autosectionlabel.*"]
