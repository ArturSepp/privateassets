"""Sphinx configuration for the PrivateAssets documentation."""

import privateassets

project = 'PrivateAssets'
author = 'Artur Sepp'
copyright = '2026, Artur Sepp'
release = privateassets.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]
exclude_patterns = ['_build']
autodoc_typehints = 'description'
html_theme = 'alabaster'
html_title = 'PrivateAssets documentation'
