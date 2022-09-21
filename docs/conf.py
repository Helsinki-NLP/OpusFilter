from pkg_resources import get_distribution

from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels import BaseLabelStyle
from pybtex.plugin import register_plugin

# General information about the project.
project = "OpusFilter"
author = "Helsinki-NLP"
language = "en"

# The full version, including alpha/beta/rc tags.
release = get_distribution('opusfilter').version
# The short X.Y version.
version = '.'.join(release.split('.')[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex"
]

source_suffix = [".rst", ".md"]
exclude_patterns = ["README.md"]

myst_heading_anchors = 2

bibtex_bibfiles = ['references.bib']
bibtex_default_style = "custom"
bibtex_reference_style = "author_year"
bibtex_cite_id = "cite-{key}"

html_theme = "sphinx_rtd_theme"
html_theme_options = {}
html_show_sourcelink = False
html_show_copyright = False
html_show_sphinx = False

suppress_warnings = ["autosectionlabel.*"]


# a simple label style which uses the bibtex keys for labels
class CustomLabelStyle(BaseLabelStyle):

    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            yield entry.key


class CustomStyle(UnsrtStyle):

    default_sorting_style = 'author_year_title'
    default_label_style = CustomLabelStyle


register_plugin('pybtex.style.formatting', 'custom', CustomStyle)
