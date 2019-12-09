import sys
import os
import strlearn
import sphinx
from sklearn.externals.six import u

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_logo = "_static/logo2.png"

html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "style_nav_header_background": "#AF2624",
    # Toc options
    # "collapse_navigation": True,
    # "sticky_navigation": True,
    # "navigation_depth": 4,
    # "includehidden": True,
    # "titles_only": False,
}

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("sphinxext"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    # "sphinx.ext.intersphinx",
    # "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    # "numpydoc",
    # "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    # "sphinx_gallery.gen_gallery",
]

numpydoc_show_class_members = False

autodoc_default_flags = ["members", "inherited-members"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# generate autosummary even if no references
autosummary_generate = True

# Generate the plots for the gallery
# plot_gallery = False

# The master toctree document.
master_doc = "index"

# General information about the project.
project = u"stream-learn"
copyright = u"2019, P. Ksieniewicz, P. Zyblewski"

version = strlearn.__version__
release = strlearn.__version__

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'sphinx'


# -- Options for HTML output ----------------------------------------------
html_static_path = ["_static"]
htmlhelp_basename = "stream-learndoc"


def setup(app):
    pass
    # app.connect('autodoc-process-docstring', generate_example_rst)
