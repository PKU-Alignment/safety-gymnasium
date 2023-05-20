# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import pathlib
import sys


ROOT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / 'safety_gymnasium'))

project = 'Safety-Gymnasium'
copyright = '2023, PKU-Alignment'
author = 'PKU-Alignment'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []


# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [('Returns', 'params_style')]

# Autodoc
autoclass_content = 'both'
autodoc_preserve_defaults = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
html_title = 'Safety-Gymnasium Documentation'
html_baseurl = 'https://www.safety-gymnasium.com'
html_copy_source = False
# html_favicon = '_static/images/favicon.png'
html_context = {
    'conf_py_path': '/docs/',
    'display_github': False,
    'github_user': 'PKU-Alignment',
    'github_repo': 'safety-gymnasium',
    'github_version': 'main',
    'slug': 'safety-gymnasium',
}

html_static_path = ['_static']
html_css_files = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
