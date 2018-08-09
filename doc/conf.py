import sys
import os
import strlearn
import sphinx

import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

sys.path.insert(0, os.path.abspath('..'))

a = strlearn.utils.arff
print(a)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'numpydoc',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery',
    'alabaster'
]

#sphinx_gallery_conf = {
    # path to your examples scripts
#    'examples_dirs' : '../examples',
    # path where to save gallery generated examples
#    'gallery_dirs'  : 'auto_examples'
#}

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        # 'donate.html',
    ]
}

# Generate the plots for the gallery
plot_gallery = False

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'stream-learn'
copyright = u'2018, Paweł Ksieniewicz'

version = strlearn.__version__
release = strlearn.__version__

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'sphinx'


# -- Options for HTML output ----------------------------------------------
#html_theme_path = [alabaster.get_path()]
#html_theme = 'alabaster'
"""
html_theme_options = {
    'logo': 'logo.png',
    'logo_name': True,
    'description': 'Python package equipped with a procedures to process data streams using estimators with API compatible with scikit-learn.',
    'github_button': True,
    'travis_button': False,
    'show_powered_by': False,
    'github_user': 'w4k2',
    'github_repo': 'stream-learn',
}
"""
html_static_path = ['_static']
htmlhelp_basename = 'stream-learndoc'


def generate_example_rst(app, what, name, obj, options, lines):
    # generate empty examples files, so that we don't get
    # inclusion errors if there are no examples for a class / module
    examples_path = os.path.join(app.srcdir, "modules", "generated",
                                 "%s.examples" % name)
    if not os.path.exists(examples_path):
        # touch file
        open(examples_path, 'w').close()


def setup(app):
    app.connect('autodoc-process-docstring', generate_example_rst)


# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'http://docs.python.org/': None}
