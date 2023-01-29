# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import pathlib
import sys
from typing import Any, Dict

ROOT_DIR = pathlib.Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / 'safety_gymnasium'))

project = 'safety_gymnasium'
copyright = '2023, pku_marl'
author = 'pku_marl'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'myst_parser',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
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
html_title = 'Safety Gymnasium Documentation'
html_baseurl = 'https://safety_gymnasium.com'
html_copy_source = False
# html_favicon = '_static/images/favicon.png'
html_theme_options = {
    # 'light_logo': 'images/logo.png',
    # 'dark_logo': 'images/logo.png',
    'gtag': 'G-6H9C8TWXZ8',
    'description': 'A standard API for reinforcement learning and a diverse set of reference environments (formerly Gym)',
    'image': 'images/logo.png',
    'versioning': True,
}
html_context: Dict[str, Any] = {}
html_context['conf_py_path'] = '/docs/'
html_context['display_github'] = False
html_context['github_user'] = 'PKU-MARL'
html_context['github_repo'] = 'Safety Gymnasium'
html_context['github_version'] = 'main'
html_context['slug'] = 'safety gymnasium'

html_static_path = ['_static']
html_css_files = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
