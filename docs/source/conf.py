# Configuration file for the Sphinx documentation builder.

# -- Project information

from strlearn import __version__

project = 'stream-learn'
copyright = u"2019, P. Ksieniewicz, P. Zyblewski"
author = u"P. Ksieniewicz, P. Zyblewski"

release = __version__
version = __version__

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "sphinxcontrib.bibtex"
]

bibtex_bibfiles = ['references_ensembles.bib', 'references_evaluators.bib']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
